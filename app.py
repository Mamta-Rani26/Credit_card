from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all 30 input values
        input_features = [float(x) for x in request.form.values()]
        
        # Convert into numpy array with correct shape
        final_features = np.array(input_features).reshape(1, -1)

        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "Fraud Transaction ❌"
        else:
            result = "Safe Transaction ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)