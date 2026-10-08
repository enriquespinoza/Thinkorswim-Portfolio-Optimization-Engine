# Schwab Primary Source Manifest

This file records the Charles Schwab Trader API documents supplied directly by the project owner and used to validate the broker-integration design.

The PDF files themselves are **not redistributed in the public repository**. The supplied copies display a Charles Schwab copyright notice and "All rights reserved." The public project therefore records source metadata, cryptographic hashes, the relevant documented interfaces, and links to the current official Schwab Developer Portal instead of republishing the full documents.

Public portal: https://developer.schwab.com/

## Source 1 — Trader API - Individual / Market Data Production

**Provenance:** project-owner supplied primary source  
**Captured:** 2026-10-08  
**Pages:** 7  
**Local file size:** 368,870 bytes  
**SHA-256:** `08b0bdb0f6ec888ac4ca2e1aa9dafa38d1f513107381976aea7ab600d7973807`

Documented production server:

```text
https://api.schwabapi.com/marketdata/v1
```

Interfaces relevant to this repository include:

```text
GET /quotes
GET /{symbol_id}/quotes
GET /chains
GET /expirationchain
GET /pricehistory
GET /movers/{symbol_id}
GET /markets
GET /markets/{market_id}
GET /instruments
GET /instruments/{cusip_id}
```

Current use in this project:

- quote retrieval for the frozen allocation universe;
- future research support for price history, option chains, market hours, and instruments;
- validation that market data are served from a distinct Schwab market-data API surface.

## Source 2 — Trader API - Individual / Accounts and Trading Production

**Provenance:** project-owner supplied primary source  
**Captured:** 2026-10-08  
**Pages:** 10  
**Local file size:** 450,573 bytes  
**SHA-256:** `a41f120719884960fdd37bf9729a86888a2dedb7cc15210c73ea4ec10022c141`

Documented production server:

```text
https://api.schwabapi.com/trader/v1
```

Read interfaces relevant to the current repository:

```text
GET /accounts/accountNumbers
GET /accounts
GET /accounts/{accountNumber}
GET /accounts/{accountNumber}/orders
GET /orders
GET /accounts/{accountNumber}/transactions
GET /accounts/{accountNumber}/transactions/{transactionId}
GET /userPreference
```

The same Schwab source also documents write-capable order interfaces:

```text
POST   /accounts/{accountNumber}/orders
DELETE /accounts/{accountNumber}/orders/{orderId}
PUT    /accounts/{accountNumber}/orders/{orderId}
POST   /accounts/{accountNumber}/previewOrder
```

Those write-capable endpoints are **not exposed by the current application boundary**. The project intentionally wraps the underlying authenticated client with `SchwabReadOnlyClient`, which exposes account discovery, account/position reads, and quotes only.

## Why hashes are recorded

The SHA-256 values identify the exact source copies used during development without requiring the public repository to redistribute the PDFs.

If Schwab changes its documentation later, a researcher can distinguish:

1. the historical source copy used to make the implementation decision;
2. the current documentation available from Schwab.

This supports reproducibility without treating third-party copyrighted documentation as project-owned material.

## Local archival convention

Developers who are permitted to retain local source copies may store them under:

```text
docs/source_materials/private/
```

That directory is gitignored and should never be committed.

Suggested local filenames:

```text
docs/source_materials/private/schwab_trader_api_market_data_production_2026-10-08.pdf
docs/source_materials/private/schwab_trader_api_accounts_trading_production_2026-10-08.pdf
```

Verify local copies with:

```bash
sha256sum docs/source_materials/private/*.pdf
```

The hashes should match the values recorded above.

## Publication rule

When discussing the Schwab integration publicly:

- cite Charles Schwab as the primary API source;
- identify the supplied documentation as project-owner supplied primary material;
- distinguish Schwab's documented capabilities from this repository's deliberately restricted read-only wrapper;
- do not imply Schwab endorses this project;
- do not redistribute the source PDFs unless the applicable terms or explicit permission allow it;
- direct readers to the current Schwab Developer Portal for the latest API terms and specifications.
