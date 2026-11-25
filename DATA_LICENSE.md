# Dataset attribution and changes

Dataset: FreshRetailNet-50K v1.0, published by Dingdong-Inc, 2025. https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

Paper: Wang et al. (2025), https://arxiv.org/abs/2505.16319

License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/legalcode

Source file: train.parquet, 106436287 bytes, SHA256 6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d, pinned at https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/08c1fab7f9257bc73679d415d65d644165d351d4/data/train.parquet

Changes made for the pilot study: `data/derived/cohort.csv` keeps 6 of 19 columns and 100 of 50,000 series (9,000 rows), selected by ascending SHA256 of `city_id,store_id,product_id`. No values were altered. Lag and rolling features are computed from it at run time. The publisher and authors do not endorse this project.

The paper abstract mentions 863 SKUs; the data card and the file contain 865 product IDs.

## Version 2.0

Second source file: eval.parquet, 8440124 bytes, SHA256 1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e, pinned at https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/08c1fab7f9257bc73679d415d65d644165d351d4/data/eval.parquet

Changes made for 2.0: `bundle/history.npz` keeps the keys, dates, `sale_amount` and `stock_hour6_22_cnt` of all 50,000 series for 35 days (2024-05-29 to 2024-07-02) as arrays. `bundle/scores.npz` adds model scores, alert flags and outcome flags for 2024-06-26 to 2024-07-02. No published value was altered. The publisher and authors do not endorse this project.
