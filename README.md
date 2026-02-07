# Smart Logistics & Supply Chain Analysis
<p align="center">

  <!-- Language & environment -->
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter Notebook">

  <!-- Data stack -->
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=plotly&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-4c72b0" alt="Seaborn">

  <!-- SQL & tools -->
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/SQL-Queries-4479A1?logo=databricks&logoColor=white" alt="SQL">
  <img src="https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white" alt="Git">
  <img src="https://img.shields.io/badge/VS%20Code-Editor-007ACC?logo=visualstudiocode&logoColor=white" alt="VS Code">

  <!-- License -->
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

## Table of Contents

- [Smart Logistics & Supply Chain Analysis](#smart-logistics--supply-chain-analysis)
- [Table of Contents](#table-of-contents)
- [1. Project Overview](#1-project-overview)
- [2. Dataset](#2-dataset)
- [3. Repository Structure](#3-repository-structure)
- [4. How to Reproduce the Analysis](#4-how-to-reproduce-the-analysis)
  - [4.1. Clone the repository](#41-clone-the-repository)
  - [4.2. Create and activate a virtual environment](#42-create-and-activate-a-virtual-environment)
  - [4.3. Install dependencies](#43-install-dependencies)
  - [4.4. Download the dataset](#44-download-the-dataset)
  - [4.5. (Optional) Build the SQLite database](#45-optional-build-the-sqlite-database)
  - [4.6. Run the notebook](#46-run-the-notebook)
- [5. Analysis Outline](#5-analysis-outline)
- [6. Technology Stack](#6-technology-stack)
- [7. Author](#7-author)
- [8. License](#8-license)


Exploratory data analysis (EDA), visualization and SQL-based investigation of a smart logistics dataset combining operational KPIs, environmental signals and user behavior data.

This project was developed as part of the Ironhack Data Science & Machine Learning Bootcamp (January 2026).

---

## 1. Project Overview

The goal of this project is to understand why a significant share of deliveries are delayed and how factors such as traffic, asset utilization, mechanical failures and waiting time influence logistics performance.

Concretely, the analysis aims to:

- Quantify the overall delay rate and on‑time performance.
- Identify which delay reasons (traffic, mechanical, weather, etc.) are most critical.
- Measure how traffic conditions and route exposure affect delays.
- Assess how asset utilization and mechanical failures impact reliability.
- Translate analytical results into actionable business recommendations.

The full narrative (business framing, analysis, and conclusions) is contained in the main notebook.

---

## 2. Dataset

The dataset used in this project is:

> Ziya (2024). *Smart Logistics & Supply Chain Dataset*. Kaggle.  
> https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset

The raw CSV file is **not** stored in this repository.

Detailed instructions to obtain and load the data are in `data/readme.md`.

High‑level data structure:

- Timestamped events for 10 trucks (Asset_ID).
- Operational variables: shipment status, inventory level, waiting time, delay flag.
- Environment: temperature, humidity, traffic status.
- Commercial behavior: transaction amount, purchase frequency.
- Logistics‑specific fields: delay reason, asset utilization, demand forecast.

---

## 3. Repository Structure

```
smart-logistics-analysis/
├── data/
│   ├── readme.md                      # How to download and load the dataset
│   ├── smart_logistics_dataset.csv    # (ignored by Git, user must download)
│   └── logistics.db                   # SQLite DB built from the CSV (local only)
├── notebooks/
│   ├── smart-logistics-analysis.ipynb # Main analysis notebook
│   ├── smart-logistics-analysis.md    # Markdown export of the notebook
│   └── smart-logistics-analysis.pdf   # PDF report of the notebook
├── sql/
│   └── logistics_queries.sql          # SQL queries used in the analysis
├── Images/
│   ├── plots/                         # All generated figures used in the report
│   └── slides/                        # Slide assets (backgrounds, cover, etc.)
├── reports/
│   ├── smart-logistics-presentation.pptx
│   └── smart-logistics-presentation.pdf
├── .gitignore
├── README.md  
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT license

```

The `.gitignore` excludes local virtual environments, Jupyter checkpoints, compiled Python files and all raw data files under `data/` (CSV, DB, Excel).

---

## 4. How to Reproduce the Analysis

### 4.1. Clone the repository

```bash
git clone https://github.com/alexandrade1978/smart-logistics-analysis.git
cd smart-logistics-analysis
```

### 4.2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
source .venv/bin/activate
```

### 4.3. Install dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, minimal stack:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels jupyter
```

### 4.4. Download the dataset

1. Go to:  
   https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset
2. Download `smart_logistics_dataset.csv`.
3. Place the file inside the `data/` directory.

For detailed instructions, see `data/readme.md`.

### 4.5. (Optional) Build the SQLite database

To run the SQL part:

1. Create `data/logistics.db`.
2. Import `smart_logistics_dataset.csv` into a table named `logistics_data`.
3. Ensure numeric and categorical columns use appropriate SQL types.
4. Run the queries in `sql/logistics_queries.sql`.

You may also adapt the notebook to generate the SQLite database programmatically.

### 4.6. Run the notebook

```bash
jupyter notebook
```

Then open:

```
notebooks/smart-logistics-analysis.ipynb
```

Run all cells from top to bottom.

---

## 5. Analysis Outline

The notebook follows a structured workflow:

1. **Initial data exploration**  
   - Load the CSV file.
   - Inspect schema, missing values, distributions and categorical labels.

2. **Operational performance and descriptive KPIs**  
   - Compute delay rates and on‑time rates across the fleet.
   - Break down performance by shipment status, delay reason and traffic status.
   - Explore waiting time and dwell‑time patterns.

3. **Driver analysis: traffic, utilization and failures**  
   - Compare delay behavior across traffic conditions and routes.
   - Analyze how asset utilization bands relate to delays and failures.
   - Identify which trucks and situations accumulate most mechanical issues.

4. **Statistical validation**  
   - Use hypothesis tests to assess whether differences in delay rates between groups are statistically significant.
   - Complement descriptive metrics with confidence intervals.

5. **Business interpretation and recommendations**  
   - Translate analytical findings into concrete actions on routing, capacity planning and maintenance.
   - Highlight which levers are likely to bring the largest improvement in on‑time delivery and cost control.

The main figures are exported to Images/plots/ from the notebook and have also been embedded into the presentation files under reports/.

---

## 6. Technology Stack

- **Language:** Python 3.13.x
- **Data analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Statistical testing:** scipy, statsmodels
- **Database:** SQLite
- **Environment:** Jupyter Notebook

---

## 7. Author

**Alexandre Andrade**  
Ironhack Data Science & Machine Learning Bootcamp (January 2026)  
GitHub: https://github.com/alexandrade1978

---

## 8. License

This project is licensed under the MIT License – see the LICENSE file for details.