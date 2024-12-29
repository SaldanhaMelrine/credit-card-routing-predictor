# Credit Card Routing Predictor

The Credit Card Routing Predictor project aims to optimize the selection of Payment Service Providers (PSPs) for routing credit card transactions. It leverages machine learning to predict transaction success rates and recommends the most cost-efficient PSP for each transaction based on predefined fee structures.

## Project Structure

- `data/`: Contains the data files ( raw transaction data, processed datasets) for PSP analysis.
- `src/`: Core Python code modules for data loading, processing, and visualization.
- `data_loader.py`: Loads data from CSV files into the database.
- `data_preprocessing.py`: Handles missing values and prepares datasets for feature engineering.
- `feature_engineering.py`: Processes data and creates new features, such as PSP success rates and cumulative fees.
- `data_visualizer.py`: Generates visualizations for model evaluation and data insights.
- `models/`:  Saved machine learning models for prediction and reuse.
- `results/`:Contains evaluation metrics, plots, and PSP recommendations.
- `README.md`: Project overview and usage guide (this file).
- `requirements.txt`: Lists Python dependencies for easy installation.

## Project Overview

This project applies machine learning to optimize credit card transaction routing. By analyzing raw transaction data and leveraging PSP-specific fee structures, it predicts transaction success probabilities, calculates routing costs, and identifies the most cost-efficient PSP for each transaction. Comprehensive visualizations and evaluation metrics ensure model reliability and transparency.

## Dependencies
Python 3.8+
Pandas: Data manipulation library.
NumPy: Numerical operations.
Matplotlib: Data visualization library for evaluation plots.
Scikit-learn: Machine learning library for training and evaluation.
Joblib: Model serialization and loading.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/credit-card-routing-predictor.git
cd credit-card-routing-predictor
pip install -r requirements.txt
