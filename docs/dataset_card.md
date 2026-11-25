# Dataset card: FreshStock 2.0 cohort

## Source
FreshRetailNet-50K, Dingdong-Inc and Wang et al. (2025), CC BY 4.0, revision 08c1fab7f9257bc73679d415d65d644165d351d4. Exact files, sizes and SHA-256 values: [DATA_LICENSE.md](../DATA_LICENSE.md).

|File|Rows|Dates|Use|
|---|---:|---|---|
|train.parquet|4,500,000|2024-03-28 to 2024-06-25|fit, tuning, calibration, thresholds|
|eval.parquet|350,000|2024-06-26 to 2024-07-02|test, scored once|

Both files hold the same 50,000 series (18 cities, 898 stores, 865 products). Six of 19 columns are used: three IDs, `dt`, `sale_amount` (normalized with an undocumented multiplier) and `stock_hour6_22_cnt` (stockout hours 06:00-22:00, 0 to 16).

## Cohort
All 50,000 series. The order rule is the SHA-256 rank of `city_id,store_id,product_id`; the first n keys give the smaller pilot cohorts, so nothing depends on outcomes. Digest f2e193a45296f3b4db4ba9d053b7e2f23873f49d557a401a874b197df2857f2d. The cohort size was chosen from measured time, memory and disk before any tuning score.

|Part|Dates (2024)|Rows|Positive|Rate|
|---|---|---:|---:|---:|
|Warm-up|28 Mar - 24 Apr|1,400,000|inputs only| |
|Fit|25 Apr - 4 Jun|2,050,000|845,053|41.22%|
|Tune|5 - 11 Jun|350,000|143,771|41.08%|
|Refit (fit + tune)|25 Apr - 11 Jun|2,400,000|988,824|41.20%|
|Calibrate|12 - 18 Jun|350,000|153,593|43.88%|
|Threshold|19 - 25 Jun|350,000|151,013|43.15%|
|Test|26 Jun - 2 Jul|350,000|143,438|40.98%|

Label: 1 if `stock_hour6_22_cnt` > 0 on the target day, else 0. Inputs use only the 28 days before it. During the test each day may use earlier test days as history, never its own day.

## Checks
Every row of both files: types, nulls, ranges, finite sales, duplicates, complete dates, and the hourly status [6:22] sum equal to the stockout count. Same keys, no date overlap, 97 consecutive days. The audit printed no test outcome. Run `python -m freshstock.expanded audit` to repeat it.

## Limits
Encoded IDs and one publisher. Sales are censored: zero sales can mean an empty shelf or no demand. No prices, weather, promotions, deliveries or stock levels are used. 97 days cover no full season. No Philippine data. The evaluation file is public, so it is fresh for this project but not secret. June 11-25 was the pilot's test period and is development data here.
