
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Default route lands on index page (the main input form)
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route receives POST form data and returns result
@app.route("/predict", methods=['POST'])
def predict():
    # Form data extraction, ensure correct mapping (reading_score, writing_score)
    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('reading_score')),    # fix mapping!
        writing_score=float(request.form.get('writing_score'))     # fix mapping!
    )
    
    pred_df = data.get_data_as_dataframe()
    print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Send prediction to index.html (or home.html if you use that as output page)
    return render_template(
        "index.html",  # or "home.html" based on your template structure
        prediction_text=f"Predicted Maths Score: {results[0]:.2f}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
