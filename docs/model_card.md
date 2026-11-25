# Model card: FreshStock 2.0, run 20261125T134353364564

All numbers come from the frozen records in [results](results/README.md) and the saved prediction rows.

## Use
- **Task:** rank 50,000 store-product series by the risk of at least one recorded stockout hour (06:00-22:00) on the next day. Supervised binary classification.
- **User:** a planner who reviews a ranked list. No automatic orders.
- **Not for:** new stores or products, other periods or retailers, demand estimation, cost or profit claims, individual explanations, or unattended decisions.

## Models
|Part|Value|
|---|---|
|Logistic Regression|Input set B, L2, C = 0.1, lbfgs, max_iter 1000, standardized numeric inputs, sigmoid calibration, alert at score >= 0.3|
|Random Forest|Input set B, 150 trees, Gini, max_depth 16, min_samples_leaf 20, sqrt features, bootstrap, seed 111, sigmoid calibration, alert at score >= 0.3; 3,426,878 nodes, 92.7 MB|
|Baseline|Persistence: alert if the series had a stockout yesterday. Not a trained model|
|Set A (11 columns)|Sales yesterday, stockout hours yesterday, 7-day mean sales, 7-day stockout days, weekday one-hot|
|Set B (18 columns)|A plus mean stockout hours 7 and 14 days, stockout days 14 and 28, mean sales 14 and 28, days since a stockout (capped at 28)|
|Software|Python 3.14.2, scikit-learn 1.8.0, numpy 2.5.5, scipy 1.18.1, pandas 2.3.3, joblib 1.6.0, pyarrow 26.0.1 (research only)|
|Code|fe1d014d (fit), protocol EXP-02 v1, lock SHA-256 b1ab1c6b...4e75, freeze SHA-256 a151d11d...16a2|

## Selection
12 settings (2 input sets x 3 per family) fitted on April 25 to June 4 and ranked by average precision on June 5 to 11. Best with set A: logistic 0.5756, forest 0.5895. Best with set B: logistic 0.5932, forest 0.6084. Differences between the three C values are about 1e-6. Refit on April 25 to June 11 (2,400,000 rows). Sigmoid fitted on June 12 to 18; used because its Brier score on June 19 to 25 was strictly lower than raw (logistic 0.213512 vs 0.213867, forest 0.211667 vs 0.211918). The forest's log loss there was slightly worse (0.611349 vs 0.611276). Thresholds: best F1 on June 19 to 25.

## Fresh test (June 26 - July 2, 2024, 350,000 rows, 143,438 positive, scored once)
|Method|AP|F1|Precision|Recall|Accuracy|Balanced acc.|TN|FP|FN|TP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Logistic Regression|0.5903|0.6218|0.4760|0.8963|0.5531|0.6055|65,007|141,555|14,869|128,569|
|Random Forest|0.6001|0.6270|0.4995|0.8420|0.5894|0.6280|85,527|121,035|22,662|120,776|
|Persistence|0.5055|0.5749|0.5768|0.5731|0.6527|0.6406|146,249|60,313|61,233|82,205|

Top 10% per day (5,000 of 50,000, 35,000 checks): stockouts found 23,144 logistic, 23,893 forest, 19,905 rule. Brier 0.215449 and 0.214183; log loss 0.619044 and 0.617438.

Both models rank better than the rule. At threshold 0.3 they flag 77.2% and 69.1% of series; their accuracy is below the rule and not above always answering no (0.5902). The rule has the best accuracy, precision and balanced accuracy. Flag-all F1 is 0.5814. No significance claim.

## Groups
After a stockout day (142,518 rows): AP 0.6499 logistic, 0.6605 forest; the forest flags all of them. After a day without one (207,482 rows): AP 0.4132 and 0.4251, rule 0.2951. Forest AP is above logistic AP in 17 of 18 cities (not in city 14). City 8 has only 455 rows.

## What the models lean on
Largest standardized logistic coefficients: stockout hours yesterday +0.4225, mean sales 7 days -0.4201, sales yesterday +0.3117. Largest forest importances: days since a stockout 0.2333, stockout hours yesterday 0.2286, stockout days 28 days 0.0973. Whole-model associations with correlated inputs. They are not causes and do not explain one prediction.

## Limits and risks
- Seven correlated days, same publisher and series. Not an independent retailer or season.
- Scores above 0.5 were higher than the observed rates on the test days. They are not exact probabilities.
- A false alert costs checking time and a miss costs an unnoticed stockout. Neither cost was measured, so the F1 thresholds may not fit a real workload.
- Fingerprints detect damage. They do not authenticate the author, and models are pickles, so only this exact package should be opened.
- Tested on one Windows laptop. Not a deployed or monitored system.
