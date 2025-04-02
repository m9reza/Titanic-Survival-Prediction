# Titanic Survival Prediction

## Overview
This project provides a machine learning solution for the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). It predicts whether a passenger survived the Titanic disaster using Random Forest and XGBoost models, combined via a simple ensemble. The approach includes feature engineering, hyperparameter tuning, and cross-validation to ensure robust performance.

## Performance
- **Cross-Validation Scores**:
  - Random Forest: 0.8395 (±0.0518)
  - XGBoost: 0.8395 (±0.0557)
- **Best Kaggle Score**: 0.77990 (public leaderboard)
- **Features Used**: `Pclass`, `Sex`, `Title`, `FamilySize`, `TicketFreq`, `Pclass_Sex`, binned `Age` and `Fare`.

## Project Structure
- `titanic.py`: Main script containing the full solution (data preprocessing, feature engineering, model training, and prediction).
- `requirements.txt`: List of required Python libraries with pinned versions.
- `train.csv`: Training data (not included, download from Kaggle).
- `test.csv`: Test data (not included, download from Kaggle).
- `submission.csv`: Output file with predictions (generated by the script).

## Requirements
- Python 3.10 or higher
- Libraries listed in `requirements.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/m9reza/token-auth-project.git
   cd token-auth-project

2. **Create a Virtual Environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run the script**:
   ```bash
   python python titanic.py