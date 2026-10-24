# Q2 2026 input refresh

The default quarter is June 30, 2026, released August 5, 2026. The Q1 configuration remains available explicitly with `--config config/2026Q1.yaml`.

Sources:

- [MFA Q2 2026 Form 10-Q](https://www.mfafinancial.com/sec-filings/all-sec-filings/content/0001055160-26-000015/mfa-20260630.htm)
- [August 5 earnings release](https://www.mfafinancial.com/news-events/press-releases/detail/365/mfa-financial-inc-announces-second-quarter-2026-financial-results)

## Input provenance

| Configuration section | Filing location | Basis |
| --- | --- | --- |
| Balance sheet | Consolidated Balance Sheets, p. 3 | $ thousands; shares in thousands; preferred values use liquidation preference, not par value |
| Loan characteristics and aging | Note 3, loan characteristics table, p. 19 | June 30 columns; exact UPB aging buckets in $ thousands; disclosed rounded LTV, FICO and 60+ rates |
| Asset allocation | Information About Our Assets, p. 64 | Seven segments; assets and financing in $ millions; debt entered as positive amounts |
| Net interest spread | Components of net interest spread, p. 72 | June 30 quarterly yield, funding cost and swap carry, converted to decimals |
| Book value bridge | GAAP to Economic Book Value reconciliation, p. 83 | $ millions; GAAP common equity 1,284.5 + loan fair value adjustment 2.4 + debt adjustment 47.5 = economic equity 1,334.4 |
| Rate sensitivity | Item 3, Interest Rate Risk, p. 88 | Reported +/-50 and +/-100bps changes, duration and convexity |
| Available securities | Item 3, Liquidity Risk, p. 90 | $294.1mm of unencumbered Agency MBS |

The loan table's delinquency buckets now use disclosed amounts. The Q1 configuration used estimated splits. The validator checks that 60+ aging agrees with each disclosed one-decimal percentage within its rounding interval.

## Modeling choices retained or clarified

- Simulation assumptions and the existing credit loss, financing and DSCR methodologies are carried forward. This refresh is not a validation of those methodologies.
- Available liquidity is unrestricted cash ($141.193mm) plus unencumbered Agency MBS ($294.1mm), subject to the selected securities haircut. At the default 15% haircut this is $391.178mm. Restricted cash ($169.029mm) is excluded. The separately disclosed $33.8mm of unencumbered whole loans is also excluded under the existing securities-only liquidity method. The allocation footnote's $56.0mm of other securities is not assumed unencumbered.
- `other_unencumbered_securities_mm` retains its existing schema name; its Q2 contents are the disclosed unencumbered Agency MBS.
- The $13,429mm allocation total is net of certain other liabilities, as in the filing's allocation table. It is not the $13,653.692mm consolidated total-assets line. Duration bucket sizes retain the existing allocation-table proxy approach: five credit segments $8,719mm, Agency MBS $4,091mm, and Other net $619mm. These are not a new exact reconstruction of MFA's duration model.
- The NPV base of $14,220mm is an approximation implied by the rounded published NPV percentages. All four published percentage changes round correctly using it. Dollar shock anchors are reproduced by the existing quadratic with coefficients -1,344.42 ($000/bp) and -4.9172 ($000/bp squared). Scenarios outside +/-100bps, including the +200bps thesis chart, are extrapolations.
- The $9.48 market price dated July 3, 2026 is retained only as a historical, user-adjustable reference. It is not presented as a current or August quote.
- The original March thesis and `MFA_Financial_Thesis.docx` remain historical. The dashboard leads with the selected quarter's facts and live charts; the original narrative is in a dated expander and its conclusions are not refreshed automatically.

## Run and verify

```powershell
python validate.py
python model.py
python -m unittest discover -s tests
streamlit run app.py
```

Defaults now select `config/2026Q2.yaml`. To inspect Q1:

```powershell
streamlit run app.py -- --config config/2026Q1.yaml
```
