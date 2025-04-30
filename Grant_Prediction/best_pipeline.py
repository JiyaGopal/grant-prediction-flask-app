import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset and specify the sheet name
file_path = r"C:\Users\user\OneDrive\Desktop\Grant_Prediction\Grant Data -  Internship jiya (8).xlsx"
tpot_data = pd.read_excel(file_path, sheet_name="Training sheet", engine="openpyxl")

# Remove whitespace from column names
tpot_data.columns = tpot_data.columns.str.strip()

# Check if the 'Status' column (target variable) exists
if 'Status' not in tpot_data.columns:  
    raise ValueError("Error: 'Status' column not found in dataset!")

# Separate features and target
features = tpot_data.drop(columns=['Status'])
target = tpot_data['Status']

# Convert categorical columns to numeric using Label Encoding
categorical_cols = features.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))  # Ensure all values are strings

# Handle missing values (fill with -1 or mean for numeric columns)
features.fillna(-1, inplace=True)
import pandas as pd

# Ensure target column has no NaNs

target = target.dropna()  # Remove NaNs

# Convert target column to string (ensure no mixed types)
target = target.astype(str)

# Check again
print("Target Unique Values:", target.unique())
print("Target Data Type:", target.dtype)
print("Features Shape:", features.shape)
print("Target Shape:", target.shape)
print("Missing values in target:", target.isnull().sum())
import pandas as pd

df = pd.read_excel(file_path)  # Load your dataset
print("Original DataFrame Shape:", df.shape)

# Drop NaNs if needed
df = df.dropna().reset_index(drop=True)
print("After NaN Removal Shape:", df.shape)

# Separate features and target
features = df.drop(columns=["target_column"])  # Replace with your actual target column name
target = df["target_column"]

print("Features Shape:", features.shape)
print("Target Shape:", target.shape)
assert features.shape[0] == target.shape[0], "Mismatch in feature and target sizes!"
import pandas as pd

df = pd.read_excel(file_path)
print("Original DataFrame Shape:", df.shape)

# Check how many NaNs exist per column
print(df.isna().sum())

# If NaNs exist, display a few affected rows
print(df[df.isna().any(axis=1)])

# Don't drop yet, check the impact first
df_no_na = df.dropna()
print("After NaN Removal Shape:", df_no_na.shape)
import pandas as pd

df = pd.read_excel(file_path)
print(df.head())  # Check first few rows
print(df.isna().sum())  # Check NaN count per column






# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
features_resampled, target_resampled = smote.fit_resample(features, target)

# Split data into training and testing sets
training_features, testing_features, training_target, testing_target = train_test_split(
    features_resampled, target_resampled, random_state=42, test_size=0.2
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(training_features, training_target)

# Make predictions
predictions = model.predict(testing_features)
print("Predictions:", predictions)
