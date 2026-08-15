"""Build and verify a deterministic, POSIX-portable reproducibility ZIP."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}
ARCHIVE_PREFIX = PurePosixPath("keldysh-finance-reproducibility")


def collect_files(source: Path) -> list[Path]:
    """Collect tracked and intentional untracked files without platform paths."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=source,
            check=True,
            capture_output=True,
        )
        candidates = [source / item.decode("utf-8")
                      for item in result.stdout.split(b"\0") if item]
    except (FileNotFoundError, subprocess.CalledProcessError):
        candidates = list(source.rglob("*"))
    return sorted(
        path for path in candidates
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(source).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def build_archive(source: Path, output: Path, files: list[Path] | None = None) -> Path:
    source = source.resolve()
    output = output.resolve()
    selected = collect_files(source) if files is None else list(files)
    selected = sorted(
        selected,
        key=lambda path: path.resolve().relative_to(source).as_posix(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in selected:
            relative = PurePosixPath(path.resolve().relative_to(source).as_posix())
            name = str(ARCHIVE_PREFIX / relative)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes())
    verify_archive(output)
    return output


def verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt member: {bad}")
        names = archive.namelist()
    if not names:
        raise ValueError("archive is empty")
    if any("\\" in name for name in names):
        raise ValueError("archive contains Windows path separators")
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
           for name in names):
        raise ValueError("archive contains unsafe paths")
    if any(PurePosixPath(name).parts[0] != str(ARCHIVE_PREFIX) for name in names):
        raise ValueError("archive members do not share the release prefix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source / "dist" / "keldysh-finance-reproducibility.zip"
    built = build_archive(args.source, output)
    print(built)
    digest = hashlib.sha256(built.read_bytes()).hexdigest()
    checksum = built.with_suffix(built.suffix + ".sha256")
    checksum.write_text(f"{digest}  {built.name}\n", encoding="ascii")
    print(checksum)


if __name__ == "__main__":
    main()
