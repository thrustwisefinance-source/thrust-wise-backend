"""
Set 3 stock scanners - additive, independent of the CANSLIM scanner modules
that live next to this package (app/scanners/*.py are not modified).

    quantitative/technical_fundamental_composite.py   Technical-Fundamental
        Composite Scoring (Stock Grade System): a 0-4 score per stock.
    services/stock_data_service.py   price history + fundamentals (yfinance,
        through the EXISTING app.scanners.yfinance_client)
    services/scoring_service.py      pure SMA / RSI / condition scoring
    services/cache_service.py        tw:set3:scanner:* cache namespace
    schemas/quantitative.py          request / response models

These are screening analytics, not trading systems: no buy / sell / hold.
"""
