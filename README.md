# ShopSpark Click Prediction with Ensemble Techniques

An end-to-end machine-learning case study that predicts whether a shopper will
click a displayed product and uses the predicted probability as a recommendation
ranking score.

The project compares conventional baselines with tuned Gradient Boosting,
AdaBoost, Random Forest, and XGBoost models. Tuned XGBoost was selected using
validation PR-AUC and evaluated once on a later test period.

## Business objective

ShopSpark currently fills recommendation slots using broad rules such as best
sellers, recent views, and discounts. This project estimates:

> The probability that a user clicks a particular product when it is displayed
> in a specific page, slot, device, and time context.

Those probabilities can rank eligible products from most to least likely to
receive a click.

## Final results

| Metric | Result |
|---|---:|
| Selected model | Tuned XGBoost |
| Validation PR-AUC | 0.2828 |
| Test PR-AUC | 0.2749 |
| Test ROC-AUC | 0.7954 |
| Test log loss | 0.2132 |
| Test Brier score | 0.0581 |
| Top-10% test CTR | 28.36% |
| Top-10% CTR lift | 4.02x |

XGBoost improved validation PR-AUC by approximately 24.2% relative to the
single decision-tree baseline. The highest-scored 10% of test impressions
captured roughly 40% of observed clicks.

See the [stakeholder HTML report](reports/stakeholder_case_study_report.html)
for the complete business summary, model comparison, risks, and recommendations.

Live report after GitHub Pages is enabled:
[namees-albayati.github.io/shopspark-click-prediction-ensembles](https://namees-albayati.github.io/shopspark-click-prediction-ensembles/)

## Workflow

1. Join impression, product, and user datasets.
2. Validate keys, schema, missing values, duplicates, and time coverage.
3. Explore CTR across user, product, placement, device, and time dimensions.
4. Create a chronological 70% training, 15% validation, and 15% test split.
5. Handle invalid values, missing data, skew, and categorical encoding.
6. Create leakage-safe historical and contextual features.
7. Establish dummy, logistic-regression, and decision-tree baselines.
8. Tune four ensemble families with parameter grids.
9. Select the winner using validation PR-AUC and evaluate it once on test data.
10. Convert click probabilities into rankings and stakeholder recommendations.

## Model comparison

| Model | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|
| **XGBoost** | **0.2828** | **0.8021** |
| Gradient Boosting | 0.2766 | 0.7967 |
| Random Forest | 0.2691 | 0.7908 |
| Decision Tree | 0.2277 | 0.7643 |
| AdaBoost | 0.2102 | 0.7645 |
| Logistic Regression | 0.1985 | 0.7566 |
| Dummy | 0.0701 | 0.5000 |

PR-AUC is the primary metric because only about 7% of impressions are clicks.
Accuracy would be misleading for this imbalanced target.

## Important engineered features

- Product historical CTR and click count
- User–product historical CTR gap
- Category and brand historical CTR
- Category–page historical CTR gap
- Time, price, discount, rating, and placement interactions
- Cold-start and user–category affinity indicators

Historical features use earlier events only. Validation and test histories are
derived from training aggregates to prevent target leakage.

## Repository structure

```text
.
├── data/                              # Local data; generated files ignored
├── docs/business_context.md
├── notebooks/
│   ├── 01_eda/
│   ├── 02_preprocessing/
│   ├── 03_feature_engineering/
│   ├── 04_modeling/
│   └── 05_reporting/
├── reports/stakeholder_case_study_report.html
├── src/
└── requirements.txt
```

## Run locally

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter lab
```

Place these files in `data/raw/`:

```text
impressions.csv
product_catalog.csv
user_profile.csv
```

Run the notebooks in numerical order. Raw data, processed data, fitted models,
and large generated CSVs are intentionally excluded from Git.

## Limitations

- Complete candidate products are not identified for each recommendation
  request, so this is offline prioritization rather than a complete reranking
  simulation.
- Offline results show predictive association, not causal CTR or revenue lift.
- Production requires candidate-set logging, business constraints, monitoring,
  and an online A/B test.
- The case narrative states 704 catalog products, while the supplied catalog
  contains 1,200 rows; this analysis reports the observed data.
