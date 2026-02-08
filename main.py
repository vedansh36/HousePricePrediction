from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load data
data = pd.read_csv("Cleaned_data.csv")

# Load model correctly
with open("RidgeModel.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get("location")
        bhk = int(request.form.get("bhk"))
        bath = int(request.form.get("bath"))
        sqft = float(request.form.get("total_sqft"))

        input_df = pd.DataFrame(
            [[location, sqft, bath, bhk]],
            columns=["location", "total_sqft", "bath", "bhk"]
        )

        prediction = model.predict(input_df)[0]

        return jsonify({"price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
