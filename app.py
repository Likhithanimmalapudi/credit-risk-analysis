from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

@app.route('/main_form.html')
def form_page():
    # Render the main form page
    return render_template('main_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect numerical features from the form
            person_age = float(request.form['person_age'])
            person_income = float(request.form['person_income'])
            loan_amnt = float(request.form['loan_amnt'])
            loan_int_rate = float(request.form['loan_int_rate'])
            loan_percent_income = float(request.form['loan_percent_income'])
            person_emp_length = float(request.form['person_emp_length'])
            cb_person_cred_hist_length = float(request.form['cb_person_cred_hist_length'])

            # Collect categorical features from the form
            person_home_ownership = request.form['person_home_ownership']
            loan_intent = request.form['loan_intent']
            loan_grade = request.form['loan_grade']
            cb_person_default_on_file = request.form['cb_person_default_on_file']

            # Combine features into an array
            features = [
                person_age, person_income, loan_amnt, loan_int_rate, loan_percent_income,
                person_emp_length, cb_person_cred_hist_length, person_home_ownership,
                loan_intent, loan_grade, cb_person_default_on_file
            ]

            # Ensure numerical and categorical features are processed
            features_array = np.array(features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(features_array)

            # Determine result based on prediction

            risk = "Low" if prediction == 0 else "High"
            sentiment = "Positive" if prediction == 0 else "Negative"

            return redirect(f"/result.html?risk={risk}&sentiment={sentiment}")


            # Redirect to result page with parameters


        except Exception as e:
            return f"An error occurred: {e}"

@app.route('/result.html')
def result_page():
    # Get prediction result from query parameters
    risk = request.args.get('risk', "Unknown")
    return render_template('result.html', risk=risk)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)