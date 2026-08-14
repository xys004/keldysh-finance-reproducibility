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
