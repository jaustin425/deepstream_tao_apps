Place hotlist source files in this directory:

- `svs.tbl` for stolen vehicles
- `slr.tbl` for stolen license plates
- `sfr.tbl` for felony vehicles

Expected file format:

```text
DATE04/07/2026 11:00
9ASX920 CA0120230108
4ZOG280 CA1920241122
8NQT723 CA1920260406
```

Parsed fields per record:

- `plate`
- `state`
- `county_code`
- `entry_date`

The backend matches by normalized uppercase `plate` only in v1, while preserving state and county metadata on each hit.