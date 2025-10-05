import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from scipy.stats import randint, uniform

# import CSV as datafile, including NSP labels
df = pd.read_csv('CTGdata.csv')

# use this function to clean data, execute below
def clean_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    result_df = result_df.dropna()

    # Calculate the difference (e - b)
    difference = result_df['e'] - result_df['b']
    
    # Drop rows where difference is zero or negative
    result_df = result_df[difference > 0]
    
    # Recalculate difference after filtering
    difference = result_df['e'] - result_df['b']
    
    # Define the columns to divide
    columns_to_divide = ['AC', 'FM', 'UC', 'DL', 'DS', 'DP']
    
    # Divide each specified column by the difference
    for col in columns_to_divide:
        if col in result_df.columns:
            result_df[col] = result_df[col] / difference

    # Drop unecessary columns
    result_df.drop(columns=['DR', 'Min', 'Max', 'Mode', 'Mean', 'b', 'e'], inplace=True)

    # Convert ASTV and ALTV to decimal percentages
    result_df['ASTV'] = result_df['ASTV']/100
    result_df['ALTV'] = result_df['ALTV']/100
    
    return result_df

clean_df = clean_data(df)
# print(clean_df.head())

# Features and target
# x  = clean_df.iloc[:,:-1]
x  = clean_df
y = df['NSP'] # use own labels in form of {1,2,3}, default taken from datafile

# print(x)
# print(y)

# Adjust labels for XGBoost (0-based)
y_xgb = y - 1
# print(y_xgb)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y_xgb, test_size=0.2, random_state=42
)

# Base model (CPU)
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)

# Parameter distribution
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 5),
    "min_child_weight": randint(1, 10)
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=200,
    scoring="f1_weighted",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit search
random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Predictions
y_pred = best_xgb.predict(X_test)

# Convert back to {1,2,3}
y_test_labels = y_test + 1
y_pred_labels = y_pred + 1

# Evaluation
print("=== Tuned CPU XGBoost Results ===")
print("F1-weighted:", f1_score(y_test_labels, y_pred_labels, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred_labels))

# Suppose your trained model is called xgb_model
best_xgb.save_model("best_xgb_model.json")  # saves in JSON format
