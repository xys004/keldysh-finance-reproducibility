from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_release_archive import build_archive


def test_release_archive_uses_safe_posix_members(tmp_path):
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    readme = source / "README.md"
    data = nested / "data.txt"
    readme.write_text("read me", encoding="utf-8")
    data.write_text("data", encoding="utf-8")
    target = tmp_path / "release.zip"

    build_archive(source, target, files=[readme, data])

    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == [
            "keldysh-finance-reproducibility/README.md",
            "keldysh-finance-reproducibility/nested/data.txt",
        ]
        assert archive.testzip() is None
