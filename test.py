import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, classification_report

# import CSV as datafile, including NSP labels
df = pd.read_csv('CTGdata.csv')

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

loaded_model = xgb.XGBClassifier()  # create a new instance
loaded_model.load_model("best_xgb_model.json")

# x_test = clean_df.iloc[:,:-1] # for me ONLY
x_test = clean_df
y_test = df['NSP']  # {1,2,3}

# Predictions
y_pred = loaded_model.predict(x_test)

# Convert back to {1,2,3}
y_pred_labels = y_pred + 1

print("F1-weighted:", f1_score(y_test, y_pred_labels, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))