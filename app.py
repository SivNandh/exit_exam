from functools import lru_cache
from math import isfinite
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("best_model.joblib")

CLASS_LABELS = {
	"Online": "Online shopping",
	"Offline": "In-store shopping",
	"Equal / both": "A mix of online and in-store",
}

FORM_SECTIONS = [
	{
		"title": "About you",
		"description": "A few basic details help put shopping preferences in context.",
		"fields": [
			{"key": "Age", "label": "Age", "kind": "number", "min": 13, "max": 100, "step": 1},
			{"key": "Gender", "label": "Gender", "options": ["Female", "Male", "Prefer not to say"]},
			{"key": "Occupation_Status", "label": "Occupation status", "options": ["Student", "Employed full-time", "Employed part-time", "Self-employed", "Unemployed"]},
			{"key": "Country_Region", "label": "Country or region", "kind": "text", "placeholder": "e.g. India"},
			{"key": "Monthly_Income_Range", "label": "Monthly income range", "options": ["Under $1,000", "$1,000-3,000", "$3,000-6,000", "$6,000+"]},
		],
	},
	{
		"title": "Shopping habits",
		"description": "Tell us where, how often, and what you usually shop for.",
		"fields": [
			{"key": "Online_Shopping_Frequency", "label": "Online shopping frequency", "options": ["Rarely", "Occasionally", "Often", "Always"]},
			{"key": "Preferred_Platform", "label": "Preferred shopping platform", "options": ["Global marketplace (Amazon etc.)", "Local e-commerce", "Brand websites", "Physical stores"]},
			{"key": "Primary_Device", "label": "Primary device", "options": ["Smartphone", "Laptop", "Tablet"]},
			{"key": "Top_Spending_Category", "label": "Top spending category", "options": ["Clothing", "Electronics", "Groceries", "Beauty & personal care", "Books", "miscellaneous"]},
			{"key": "Makes_Shopping_List", "label": "Makes a shopping list", "options": ["Never", "Sometimes", "Often", "Always"]},
			{"key": "Impulse_Purchase_Frequency", "label": "Impulse purchase frequency (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Return_Frequency", "label": "How often do you return purchases?", "options": ["Never", "Sometimes", "Often"]},
			{"key": "Avg_Monthly_Spend_NonEssentials", "label": "Monthly non-essential spending", "options": ["Under $50", "$50-150", "$150-300", "$300-500", "$500+"]},
		],
	},
	{
		"title": "Reviews, ads and payment",
		"description": "Rate each item using its displayed scale. You can leave anything blank.",
		"fields": [
			{"key": "Review_Influence_Score", "label": "Influence of reviews (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Social_Ads_Influence_Score", "label": "Influence of social ads (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Price_Comparison_Frequency", "label": "Price comparison frequency (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Discount_Importance_Score", "label": "Importance of discounts (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Follows_Brands_On_Social", "label": "Follows brands on social media", "options": ["Yes", "No"]},
			{"key": "Likely_To_Buy_After_Ad_Score", "label": "Likelihood to buy after an ad (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
			{"key": "Preferred_Payment_Method", "label": "Preferred payment method", "options": ["Cash", "Debit Card", "Credit Card", "Digital wallet"]},
			{"key": "Uses_BNPL_Installments", "label": "Uses buy-now-pay-later installments", "options": ["No, never", "Yes, occasionally", "Yes, regularly"]},
			{"key": "Online_Shopping_Satisfaction_Score", "label": "Online shopping satisfaction (1-10)", "kind": "number", "min": 1, "max": 10, "step": 1},
			{"key": "Trust_In_Online_Reviews_Score", "label": "Trust in online reviews (1-10)", "kind": "number", "min": 1, "max": 10, "step": 1},
			{"key": "Regret_After_Purchase_Frequency", "label": "Regret after purchase (1-5)", "kind": "number", "min": 1, "max": 5, "step": 1},
		],
	},
]

FEATURES = [field["key"] for section in FORM_SECTIONS for field in section["fields"]]
NUMERIC_FIELDS = {
	field["key"]: (field["min"], field["max"])
	for section in FORM_SECTIONS
	for field in section["fields"]
	if field.get("kind") == "number"
}


@lru_cache(maxsize=1)
def load_model():
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
	model = joblib.load(MODEL_PATH)
	if not callable(getattr(model, "predict", None)):
		raise TypeError("The saved model does not provide a predict method.")
	return model


def build_prediction_input(form):
	values = {}
	errors = []
	entered_count = 0

	for field in FEATURES:
		raw_value = form.get(field, "").strip()
		if not raw_value:
			values[field] = float("nan")
			continue

		entered_count += 1
		if field in NUMERIC_FIELDS:
			try:
				value = float(raw_value)
			except ValueError:
				errors.append(f"{field.replace('_', ' ')} must be a number.")
				continue

			low, high = NUMERIC_FIELDS[field]
			if not isfinite(value) or not low <= value <= high:
				errors.append(
					f"{field.replace('_', ' ')} must be between {low} and {high}."
				)
				continue
			values[field] = value
		else:
			values[field] = raw_value

	if entered_count < 3:
		errors.append("Please enter at least three shopping details to get a prediction.")

	return pd.DataFrame([values], columns=FEATURES), errors


@app.get("/")
def home():
	return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
	prediction = None
	error = None
	form_values = {}

	if request.method == "POST":
		form_values = request.form.to_dict()
		model_input, errors = build_prediction_input(request.form)
		if errors:
			error = " ".join(errors)
		else:
			try:
				predicted_class = str(load_model().predict(model_input)[0])
				prediction = {
					"label": CLASS_LABELS.get(predicted_class, predicted_class),
					"class": predicted_class,
				}
			except Exception:
				app.logger.exception("Shopping mode prediction failed")
				error = (
					"The saved model could not process these details. Check that "
					"best_model.joblib is the fitted preprocessing-and-model pipeline "
					"trained with these feature names."
				)

	return render_template(
		"predict.html",
		sections=FORM_SECTIONS,
		prediction=prediction,
		error=error,
		form_values=form_values,
	)


if __name__ == "__main__":
	app.run(debug=True)
