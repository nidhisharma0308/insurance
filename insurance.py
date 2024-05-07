from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Import encoders


app = Flask(__name__)
model = pickle.load(open('insurance.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('home.html')

def value_pred(to_pred_list):
    to_pred= np.array(to_pred_list).reshape(-1,1)
    model = pickle.load(open('insurance.pkl', 'rb'))
    result=model.predict(to_pred)
    return result[0]



@app.route('/submit', methods=['POST'])
def submit():
    age = float(request.form['age'])
    

    # Example for sex
    sex_encoder = LabelEncoder()
    sex_encoded = sex_encoder.fit_transform([request.form['sex']])[0]  # Encode and get the first element (label)

    # Repeat for other categorical features

    bmi = float(request.form['bmi'])
    
    children = float(request.form['children'])

    smoker_encoder=LabelEncoder()
    smoker_encoded = smoker_encoder.fit_transform([request.form['smoker']])[0]
    region_encoder= LabelEncoder()
    region_encoded = region_encoder.fit_transform([request.form['region']])[0]

    arr = np.array([age, sex_encoded, bmi, children, smoker_encoded, region_encoded]).reshape(1, -1)

    prediction = model.predict(arr)

    return jsonify({'prediction': prediction[0]})  # Return prediction in JSON format

if __name__ == '__main__':
    app.run(debug=True)
