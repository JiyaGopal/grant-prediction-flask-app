import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import joblib

file_path = "C:\\Users\\user\\OneDrive\\Desktop\\Grant_Prediction\\Grant Data -  Internship jiya (8).xlsx"
df = pd.read_excel(file_path, sheet_name="Training sheet", engine='openpyxl')
df.columns = df.columns.str.strip()

drop_columns = ["Unique ID", "Startup Name", "Founder Name", "Name of the Incubator", 
                "Status", "If received then grant type"]
df = df.drop(columns=drop_columns, errors='ignore')

df = df.dropna(subset=["Grant Applied"])

num_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns.drop("Grant Applied")
df[cat_cols] = df[cat_cols].fillna("Unknown")

for col in cat_cols:
    df[col] = df[col].str.strip().str.lower()

joblib.dump(cat_cols.tolist(), "categorical_columns.pkl")
joblib.dump(num_cols.tolist(), "numeric_columns.pkl")

multi_label_target = pd.get_dummies(df["Grant Applied"])
df_multi_label = pd.concat([df.drop(columns=["Grant Applied"]), multi_label_target], axis=1)

df_grouped = df_multi_label.groupby(list(df.drop(columns=["Grant Applied"]).columns)).max().reset_index()

cat_features = df_grouped.select_dtypes(include="object").columns

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_grouped[cat_features] = ordinal_encoder.fit_transform(df_grouped[cat_features])

grant_cols = multi_label_target.columns
joblib.dump(grant_cols.tolist(), "grant_labels.pkl")
X = df_grouped.drop(columns=grant_cols)
y = df_grouped[grant_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = HistGradientBoostingClassifier(random_state=42)
multi_model = MultiOutputClassifier(base_model)

param_grid = {
    'estimator__max_iter': [100, 200],
    'estimator__learning_rate': [0.05, 0.1],
    'estimator__max_leaf_nodes': [31, 63]
}

grid_search = GridSearchCV(multi_model, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

joblib.dump(best_model, "best_model.pkl")
joblib.dump(ordinal_encoder, "ordinal_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("Best Parameters:", grid_search.best_params_)  
y_pred = best_model.predict(X_test)
print("\nMulti-label accuracy (average of all grants):", round(best_model.score(X_test, y_test) * 100, 2), "%")

for i, grant in enumerate(grant_cols):
    print(f"\nGrant: {grant}")         
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

new_startup_data = pd.DataFrame([{
    'Product name': 'kerala',
    'Product Stage': 'Ideation & Designing Stage',
    'Industry': 'Healthcare',
    'Technology Domain': 'ML',
    'Are you a Student Startup?': 'Yes', 
    "Are you a Women's Startup?": 'yes',
    'Are you a Transgender Startup?': 'No',
    'Details of previous grants received from KSUM': 'no',
    'Amount': 0,
    'Business Model': 'Business to Business',
    'Details of fund raised in last 2 years (Sum)': 1000000,
    'Investments made so far including own funds': 2000000,  
    'Monthly turnover': 100000
}])

raw_startup_data = new_startup_data.copy()

for col in cat_cols:
    new_startup_data[col] = new_startup_data[col].str.strip().str.lower()

new_startup_data[cat_features] = ordinal_encoder.transform(new_startup_data[cat_features])
new_startup_data = new_startup_data[X.columns]  

predicted_grants = best_model.predict(new_startup_data)
eligible_grants = list(grant_cols[predicted_grants[0] == 1])

if raw_startup_data.iloc[0]["Are you a Women's Startup?"].strip().lower() == "yes":
    eligible_grants = list(set(eligible_grants + ["Women Productisation Grant"]))

if raw_startup_data.iloc[0]["Are you a Student Startup?"].strip().lower() == "yes":
    eligible_grants = list(set(eligible_grants + ["Student Innovation Grant"]))

if eligible_grants: 
    print("\nFinal predicted eligible grants for this startup:")                                             
    for grant in eligible_grants:
        print(" -", grant)
else: 
    print("\nModel did not predict any eligible grants")

