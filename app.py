import pickle
from flask import Flask, request, render_template, abort

# Load the model (pipeline recommended with scaler/encoder)
model = pickle.load(open('model_pickle', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", result='')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # For now, just print the contact message to the console
        print(f"Message from {name} ({email}): {message}")

        return render_template("contact.html", success="Your message has been sent successfully!")

    except KeyError as e:
        abort(400, f"Missing form field: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Required numeric inputs
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        heart_diseases = float(request.form['heart_diseases'])
        smoking_history = float(request.form['smoking_history'])

        # --- Compute BMI ---
        if 'height_cm' in request.form and request.form['height_cm'].strip():
            height_cm = float(request.form['height_cm'])
            weight_kg = float(request.form['weight_kg'])
            height_m = height_cm / 100.0
            bmi = weight_kg / (height_m ** 2)
        elif 'height_m' in request.form and request.form['height_m'].strip():
            height_m = float(request.form['height_m'])
            weight_kg = float(request.form['weight_kg'])
            bmi = weight_kg / (height_m ** 2)
        else:
            bmi = float(request.form['bmi'])

        hba1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        # Arrange inputs in training order
        input_data = [[
            gender,
            age,
            hypertension,
            heart_diseases,
            smoking_history,
            bmi,
            hba1c_level,
            blood_glucose_level
        ]]

        # Get probability
        prob = model.predict_proba(input_data)[0][1]
        risk_percent = round(prob * 100, 2)

        # Get class prediction
        prediction = model.predict(input_data)[0]
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        return render_template(
            'index.html',
            result=f"{risk_label} ({risk_percent}% probability)"
        )

    except (KeyError, ValueError) as e:
        abort(400, f"Invalid or missing input: {e}")


if __name__ == "__main__":
    app.run(debug=True)
