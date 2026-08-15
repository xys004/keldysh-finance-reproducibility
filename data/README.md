# Data provenance

The empirical analysis uses public Binance candlestick data for BTCUSDT,
ETHUSDT, BNBUSDT, and SOLUSDT at hourly and four-hourly resolution. The
signed flow is computed from the taker-buy volume and total volume fields of
each candle. No private order-book feed or proprietary trade archive is
required.

The downloader in `src/keldysh_finance/flow.py` stores a local cache only for
the duration of a run. The cache is ignored by Git because public market data
can change through corrections and because the archived JSON outputs are the
versioned evidence associated with the manuscript. A reproducible release
should record the retrieval date, request parameters, and checksum of any
local raw-data snapshot used to regenerate a figure.

Experiment 13 additionally downloads one complete UTC day of public Binance
Vision `trades` archives for BTCUSDT and ETHUSDT. These archives are used only
to calibrate the transfer-size normalisation and to reconcile hourly fields;
they are cached locally under `output/cache/trades/` and are not redistributed.
The archived JSON records the date, sample counts, calibration ratios, and
reconciliation errors.
