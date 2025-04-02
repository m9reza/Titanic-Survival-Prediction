# Import necessary libraries
# - numpy and pandas for data manipulation and numerical operations
# - RandomForestClassifier and XGBClassifier for building machine learning models
# - cross_val_score and GridSearchCV for model evaluation and hyperparameter optimization
# - StandardScaler for feature scaling to ensure consistent model performance
# - SimpleImputer for handling missing values in the dataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
# - train.csv contains the training data with features (e.g., Pclass, Sex) and target ('Survived')
# - test.csv contains the test data for which we need to predict 'Survived'
# - These files are standard for the Kaggle Titanic competition and must be in the working directory
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Feature Engineering Function
# - This function processes the raw data and creates new features to improve model performance
# - Why: Feature engineering captures domain-specific patterns (e.g., social status, family dynamics) that are critical for predicting survival
def process_data(df):
    # Create FamilySize by summing SibSp (siblings/spouses) and Parch (parents/children) plus 1 (the passenger themselves)
    # Why: FamilySize indicates whether a passenger traveled alone or with family, which can influence survival odds (e.g., families might be prioritized for lifeboats)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create IsAlone feature: 1 if FamilySize is 1, 0 otherwise
    # Why: Solo travelers might have different survival chances (e.g., more mobility but less priority in rescue efforts)
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Create TicketFreq by counting how many passengers share the same ticket number
    # Why: TicketFreq can indicate group travel (e.g., families or friends), which might correlate with survival (e.g., groups might help each other)
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')

    # Create Pclass_Sex interaction feature by combining Pclass and Sex into a single categorical feature
    # Why: Survival rates vary significantly by class and gender (e.g., 1st-class females had very high survival rates), and this interaction captures that combined effect
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']

    # Extract Title from Name using regex to find the word before a period (e.g., Mr, Mrs, Miss)
    # Why: Titles reflect social status, age, and gender, which are strong predictors of survival (e.g., 'Mr' often indicates adult males with lower survival rates)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Group rare titles into a 'Rare' category and standardize similar titles (e.g., Mlle to Miss)
    # Why: Reduces noise by consolidating infrequent titles while preserving meaningful categories (e.g., 'Miss' for young females)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Bin Age into categories to capture nonlinear effects
    # Why: Binning helps the model generalize better by grouping ages into meaningful categories (e.g., children had higher survival rates)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
                          labels=['Infant', 'Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAge', 'Senior', 'Elder'])

    # Bin Fare into 5 quantiles to capture economic status
    # Why: Fare reflects wealth, which correlates with survival (higher fares often mean higher class and better survival odds)
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])

    return df


# Apply feature engineering to both train and test data
# Why: Ensures consistency in feature creation between train and test sets, which is necessary for accurate predictions
train_data = process_data(train_data)
test_data = process_data(test_data)

# Select features for modeling
# - Includes original features (Pclass, Sex, etc.) and engineered features (FamilySize, TicketFreq, etc.)
# Why: These features capture the key patterns in the Titanic dataset, such as class, gender, and family dynamics, which are critical for survival prediction
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize',
            'IsAlone', 'TicketFreq', 'Pclass_Sex', 'Title', 'AgeBin', 'FareBin']

# Prepare training data
# - Convert categorical features to dummy variables (one-hot encoding) to make them compatible with machine learning models
# - Extract the target variable 'Survived' from the training data
# Why: Machine learning models require numerical inputs, and one-hot encoding preserves categorical information without assuming ordinality
X = pd.get_dummies(train_data[features])
y = train_data['Survived']
X_test = pd.get_dummies(test_data[features])

# Align test and train data to ensure they have the same columns
# - Fills missing columns in the test set with 0 (e.g., if a category present in train is missing in test)
# Why: Prevents errors during prediction if the test data lacks certain dummy variables (e.g., a rare title not present in test)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Handle missing values using SimpleImputer
# - Imputes missing values with the most frequent value in each column
# Why: Ensures there are no missing values in the dataset, which would otherwise cause errors during model training
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale numerical features using StandardScaler
# - Standardizes features to have mean 0 and variance 1
# Why: Improves model performance by ensuring all features are on the same scale, which can help tree-based models converge more effectively
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Tune Random Forest with GridSearchCV
# - Searches over a grid of hyperparameters to find the best combination for Random Forest
# - Parameters: number of trees (n_estimators), max depth, min samples for splits/leaves
# - Updated min_samples_split to ensure all values are >= 2, avoiding the InvalidParameterError
# Why: Optimizes the Random Forest model to balance bias and variance, improving its ability to generalize to the test set
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'min_samples_split': [2, 3, 5],  # Adjusted to include 3 (best from output) and ensure all values are valid
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_scaled, y)
print("Best Random Forest Parameters:", grid_search_rf.best_params_)
print("Best Random Forest CV Score:", grid_search_rf.best_score_)

# Tune XGBoost with GridSearchCV
# - Searches over a grid of hyperparameters for XGBoost
# - Parameters: max depth, number of trees (n_estimators), learning rate
# - Updated learning_rate to include 0.09 (best from output) for better tuning
# Why: Optimizes XGBoost to complement Random Forest, capturing different patterns in the data (e.g., gradient boosting vs. bagging)
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.09]  # Added 0.09 as it was the best in the output
}
xgb = XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_scaled, y)
print("Best XGBoost Parameters:", grid_search_xgb.best_params_)
print("Best XGBoost CV Score:", grid_search_xgb.best_score_)

# Cross-validation for best models
# - Evaluates the tuned Random Forest and XGBoost models using 5-fold cross-validation
# Why: Provides a robust estimate of model performance on unseen data, helping to detect overfitting and ensure the model generalizes well
rf_best = grid_search_rf.best_estimator_
xgb_best = grid_search_xgb.best_estimator_
rf_cv = cross_val_score(rf_best, X_scaled, y, cv=5, scoring='accuracy')
xgb_cv = cross_val_score(xgb_best, X_scaled, y, cv=5, scoring='accuracy')
print(f"Random Forest CV Score: {rf_cv.mean():.4f} (+/- {rf_cv.std() * 2:.4f})")
print(f"XGBoost CV Score: {xgb_cv.mean():.4f} (+/- {xgb_cv.std() * 2:.4f})")

# Ensemble: Average predictions from Random Forest and XGBoost
# - Predicts probabilities for the test set using both models
# - Averages the probabilities and applies a threshold of 0.5 to get binary predictions (0 or 1)
# Why: Combining predictions from two models reduces variance and leverages their complementary strengths (Random Forest's bagging and XGBoost's boosting)
rf_probs = rf_best.predict_proba(X_test_scaled)[:, 1]
xgb_probs = xgb_best.predict_proba(X_test_scaled)[:, 1]
ensemble_probs = (rf_probs + xgb_probs) / 2
predictions = (ensemble_probs >= 0.5).astype(int)

# Feature importance from Random Forest
# - Extracts the importance of each feature from the Random Forest model
# Why: Helps understand which features are driving the model's predictions, providing insights into the data and model behavior
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance.head(10))

# Create submission file
# - Saves the predictions in the format required by Kaggle: a CSV with 'PassengerId' and 'Survived' columns
# Why: Allows submission to Kaggle for evaluation on the test set, providing a public leaderboard score
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")