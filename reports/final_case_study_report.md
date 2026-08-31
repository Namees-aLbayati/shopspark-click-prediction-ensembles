# ShopSpark Click Prediction — Final Report

## Objective
Predict the probability that a user clicks a displayed product, then use that probability to prioritize products in recommendation slots.

## Data and validation design
The project used 450,000 impressions with a chronological 70% training, 15% validation, and 15% test split. The test period remained unavailable during model selection.

## Selected model
The selected model is **xgboost**, chosen using validation PR-AUC. Its validation PR-AUC was **0.2825**, a **24.05% relative improvement** over the single decision tree.

## Final test performance
- PR-AUC: **0.2740**
- ROC-AUC: **0.7951**
- Log loss: **0.2133**
- Brier score: **0.0581**

## Ranking result
The full test population had **7.06% CTR**. The highest-scored 10% achieved **28.31% CTR**, representing **4.01× lift** and capturing **40.11% of all test clicks**.

## Recommendation
Use the predicted click probability as the offline product-ranking score. Before deployment, add request IDs and complete eligible candidate sets, calibrate probabilities if needed, apply business constraints, and run an online A/B test against the current recommendation rules.

## Limitations
This is an observational offline evaluation. It demonstrates ranking quality but cannot prove causal CTR or revenue lift. The current data does not identify products competing in the same recommendation request.
