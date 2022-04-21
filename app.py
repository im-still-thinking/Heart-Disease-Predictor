# Imports
from flask import Flask, request, render_template, url_for, flash, redirect
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

#Initialize the Flask App
app = Flask('cleanheart')

#Loads pre - trained model
heart_model = pickle.load(open('/Users/aritrar/cleanheart/Models/heart_model.pkl', 'rb'))

#Implementing Interaction
cols = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    label_encoder = preprocessing.LabelEncoder()

    val = [i for i in request.form.values()]
    res = np.array(val)
    new_data = pd.DataFrame([res], columns = cols)
    new_data = pd.get_dummies(new_data, columns = ['Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','PhysicalActivity','Asthma','KidneyDisease','SkinCancer'])

    new_data['AgeCategory'] = label_encoder.fit_transform(new_data['AgeCategory'])
    new_data['Race'] = label_encoder.fit_transform(new_data['Race'])
    new_data['Diabetic'] = label_encoder.fit_transform(new_data['Diabetic'])
    new_data['GenHealth'] = label_encoder.fit_transform(new_data['GenHealth'])
    new_data['HeartDisease'] = label_encoder.fit_transform(new_data['HeartDisease'])

    prediction = heart_model.predict(new_data)
    if prediction == 1:
        ans = 'You have heart disease'
    elif prediction == 0:
        ans = "You don't have heart disease"

    return render_template('index.html', prediction_ans = ans)

if __name__ == '__main__':
    app.run(debug=True)

