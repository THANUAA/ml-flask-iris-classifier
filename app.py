from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and label encoder
model = pickle.load(open("iris_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to floats
        input_features = [float(x) for x in request.form.values()]
        final_features = np.array([input_features])
        
        # Predict
        prediction = model.predict(final_features)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html",
                               prediction_text=f"🌸 Predicted Iris Species: {predicted_class}")
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
