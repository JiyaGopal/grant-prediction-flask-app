from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("best_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
numeric_columns = joblib.load("numeric_columns.pkl")
categorical_features = joblib.load("categorical_columns.pkl")
grant_labels = joblib.load("grant_labels.pkl")  

@app.route("/", methods=["GET", "POST"])
def predict():
    grants = None
    if request.method == "POST":
        form_data = {
            "Product name": request.form["product_name"],
            "Product Stage": request.form["product_stage"],
            "Industry": request.form["industry"],
            "Technology Domain": request.form["technology_domain"],
            "Are you a Student Startup?": request.form["are_you_a_student_startup"],
            "Are you a Women's Startup?": request.form["are_you_a_womens_startup"],
            "Are you a Transgender Startup?": request.form["are_you_a_transgender_startup"],
            "Details of previous grants received from KSUM": request.form["previous_grants_received_from_ksum"],
            "Amount": float(request.form["amount"]),
            "Business Model": request.form["business_model"],
            "Details of fund raised in last 2 years (Sum)": float(request.form["fund_raised_last_2_years"] or 0),
            "Investments made so far including own funds": float(request.form["investments_made_so_far"]),
            "Monthly turnover": float(request.form["monthly_turnover"] or 0)
        }

        raw_df = pd.DataFrame([form_data])

        for col in categorical_features:
            raw_df[col] = raw_df[col].astype(str).str.strip().str.lower()

        raw_df[categorical_features] = encoder.transform(raw_df[categorical_features])

        raw_df = raw_df[feature_columns]

        predictions = model.predict(raw_df)[0] 
        grants = [label for label, pred in zip(grant_labels, predictions) if pred == 1]

        if request.form["are_you_a_womens_startup"].strip().lower() == "yes":
            grants.append("Women Productisation Grant")
        if request.form["are_you_a_student_startup"].strip().lower() == "yes":
            grants.append("Student Innovation Grant")

    return render_template("index.html", grants=grants)

if __name__ == "__main__":
    app.run(debug=True)
