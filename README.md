# Los Angeles Car Crash Analysis

## Background

After completing a [supervised machine learning course](https://www.coursera.org/learn/machine-learning) from DeepLearning.AI and Stanford, I wanted to put what I had learned to the test with a real-world dataset. Having lived in Los Angeles for the past two years, I had noticed the high volume of car crashes that occur throughout the city. While searching through the [Los Angeles City Open Data Catalog](https://data.lacity.org/browse), I came across a car crash database that had been actively recorded for the past 15 years.

Even though the dataset was larger and more complex than I had originally planned for, it presented an exciting challenge. Sifting through the data, I was struck by how frequently serious injury-causing crashes occurred. I started to think about how cities and police departments actively work to combat and reduce these incidents. Having learned the difference between drawing conclusions from raw frequency counts versus using logistic regression to produce a more comprehensive analysis, I decided to pursue the question:

**"Can logistic regression accurately predict whether a car crash results in an injury?"**

By building a model that actively adjusts and surfaces the statistical weight of individual factors, it becomes possible to track how different variables have influenced crash severity over time, and potentially identify patterns that cities could act on in the future.

The analysis and interactive visualization live at: [la-carcrash.com](https://www.la-carcrash.com)

---

## Attribution — Claude Code

The code in this repository was built in collaboration with [Claude Code](https://claude.ai/code), Anthropic's AI coding assistant. The vast majority of the Python implementation was written by Claude. I provided the mathematical and statistical foundation from my supervised machine learning coursework, including the logistic regression model structure, sigmoid activation function, gradient descent optimizer, cost function, and feature scaling. I understood what I wanted to build and gave Claude Code the direction.

**What I directed and oversaw:**
- Framing the core research question and deciding what was worth investigating
- Every architectural and design decision for the analysis pipeline and the website
- Extensive data analysis throughout the process — identifying outliers, recognizing anomalies, adjusting parameters, and re-running analysis until results were trustworthy
- Ensuring the findings were presented honestly and without bias toward a particular conclusion, including actively disclosing model limitations and cases where the data was ambiguous
- Validating results multiple times before treating them as findings
- Domain intuition about Los Angeles — knowing what the findings meant in real-world context, not just as numbers
- All product and UX direction for the visualization website — what to show, how to frame findings for a general audience, and what to cut

Claude Code handled the implementation and code creation. The research questions, analytical judgment, integrity of the findings, and project direction were mine.

---

## Repository Files

This repository contains the core Python pipeline used to collect, clean, train, and precompute results from the LA collision dataset — roughly 621,000 records spanning 2010 to 2024.

---

## Pipeline — How the Files Connect

Run in this order:

| Step | File | What it does |
|------|------|--------------|
| 1 | `collision_pull_yearly.py` | Fetches raw collision data year by year from the LA City open data API |
| 2 | `collision_utils.py` | Shared cleaning, feature engineering, and model math (sigmoid, gradient descent, cost function) used across all other scripts |
| 3 | `collision_model.py` | Combines all yearly CSVs, engineers features, scales the data, and saves the feature list |
| 4 | `collision_train.py` | Trains the logistic regression model on the full scaled dataset |
| 5 | `collision_retrain.py` | Retrains the model on a specific year range — used to iterate and refine |
| 6 | `collision_precompute.py` | Interactive script to train and evaluate a single custom window (choose your own start year, end year, and test year) |
| 7 | `collisions_sliding_window.py` | Runs the full sliding window analysis across all valid year combinations and saves results to `sliding_window_results_v3.json` |
| 8 | `collision_to_geojson.py` | Converts yearly CSVs into GeoJSON files for the interactive map on the website |

---

## Key Output Files

- **`sliding_window_results_v3.json`** — Precomputed results for all sliding window combinations. This is the data powering the Prediction Model page on the website.
- **`features.json`** — The feature list used across all model training runs.
- **`MO_CODES_Numerical_20180627.pdf`** — LAPD MO code reference document. MO codes are how the raw data records collision type, injury status, and circumstances.

---

## Data

The raw CSV files are not included in this repository due to size. You can download them directly from the LA City Open Data portal:

[https://data.lacity.org/Transportation/Traffic-Collision-Data-from-2010-to-Present/d5tf-ez2w](https://data.lacity.org/Transportation/Traffic-Collision-Data-from-2010-to-Present/d5tf-ez2w)

After downloading, name the files `la_collisions_<year>.csv` (e.g. `la_collisions_2019.csv`) and place them in the same directory as the scripts before running.

---

I appreciate you viewing my work. Please let me know if you have any thoughts, feedback, critiques, or questions.

Best,
J.P. Frimel
