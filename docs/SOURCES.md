# Sources and provenance

## Market observations

Production public REST endpoint base: `https://www.deribit.com/api/v2/public/`. No account or API key is used. Only public instrument metadata and market quotes are requested.

- [Instrument metadata](https://docs.deribit.com/api-reference/market-data/public-get_instruments)
- [Currency book summaries](https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_currency)
- [Selected instrument order books](https://docs.deribit.com/api-reference/market-data/public-get_order_book)
- [Inverse option contract conventions](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options)
- [USD option quote convention](https://support.deribit.com/hc/en-us/articles/25944775983133-USD-Order-Options)

`data/manifest.json` records source URLs, UTC capture times, selection exclusions, conventions and a SHA-256 digest for every response. The selected capture is also archived under `data/raw/<capture_id>/`. `data/processed/quotes.json` and `rejected_quotes.json` are regenerated from verified raw responses. Sources retain their own terms; observed public availability is not a claim of ownership of exchange data.

The inspected instrument lists contain 944 BTC and 830 ETH contracts. The bounded capture selects 44 per currency. All 88 selected order-book responses passed post-capture cleaning in this snapshot; that does not mean the entire listed universe passed the selection filters. There are no historical option series in this project.

## Model references

- Jim Gatheral and Antoine Jacquier, [Arbitrage-free SVI volatility surfaces](https://arxiv.org/abs/1204.0646), revised 2013. SSVI parameterization, density criterion and sufficient static-arbitrage restrictions; the implementation uses the modified power-law family with exponent 1/2.
- Steven L. Heston, [A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options](https://doi.org/10.1093/rfs/6.2.327), Review of Financial Studies 6(2), 1993. Model dynamics and characteristic-function pricing.

Implementation is independently written with Codex assistance. No calibration quality, live execution, affiliation, professional review, or personal sole-authorship claim is inferred from these references.
