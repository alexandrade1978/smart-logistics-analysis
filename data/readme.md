# Data

The dataset used in this project is:

> Ziya (2024). *Smart Logistics & Supply Chain Dataset*. Kaggle.  
> https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset

The raw CSV file is **not** stored in this repository.  
To reproduce the analysis and SQL queries:

1. Download `smart_logistics_dataset.csv` from Kaggle.  
2. Create a SQLite database and import the CSV into a table named `logistics_data`.  
3. Ensure columns are imported with appropriate types (INTEGER / REAL / TEXT).  
4. Run the queries provided in `sql/logistics_queries.sql`.