# Imports
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

#Initialize the Flask App
app = Flask(__name__, template_folder = 'templates')

#Loads pre - trained model
heart_model = pickle.load(open('heart_model.pkl', 'rb'))

#Implementing Interaction
cols1 = [
 'BMI',
 'Smoking',
 'AlcoholDrinking',
 'Stroke',
 'PhysicalHealth',
 'MentalHealth',
 'DiffWalking',
 'Diabetic',
 'PhysicalActivity',
 'GenHealth',
 'SleepTime',
 'Asthma',
 'KidneyDisease',
 'SkinCancer',
 'Sex_Female',
 'Sex_Male',
 'AgeCategory_18-24',
 'AgeCategory_25-29',
 'AgeCategory_30-34',
 'AgeCategory_35-39',
 'AgeCategory_40-44',
 'AgeCategory_45-49',
 'AgeCategory_50-54',
 'AgeCategory_55-59',
 'AgeCategory_60-64',
 'AgeCategory_65-69',
 'AgeCategory_70-74',
 'AgeCategory_75-79',
 'AgeCategory_80 or older',
 'Race_American Indian/Alaskan Native',
 'Race_Asian',
 'Race_Black',
 'Race_Hispanic',
 'Race_Other',
 'Race_White']

cols2 = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():

    val1 = [0 for _ in range(35)]
    val2 = [i for i in request.form.values()]

    new_data1 = pd.DataFrame(dict(zip(cols1, val1)), columns = cols1, index = [0])
    new_data2 = pd.DataFrame(dict(zip(cols2, val2)), columns = cols2, index = [0])

    one_hot = ['Sex', 'AgeCategory', 'Race']
    new_data2 = pd.get_dummies(new_data2, columns = one_hot)
    new_data1.loc[:, list(new_data1.columns)] = new_data2.loc[:, list(new_data2.columns)]
    new_data1 = new_data1.fillna(0)

    new_data1 = new_data1.astype({'BMI' : float, 'Smoking' : int,
 'AlcoholDrinking' : int,
 'Stroke' : int,
 'PhysicalHealth' : int,
 'MentalHealth' : int,
 'DiffWalking' : int,
 'Diabetic' : int,
 'PhysicalActivity' : int,
 'GenHealth' : int,
 'SleepTime' : int,
 'Asthma' : int,
 'KidneyDisease' : int,
 'SkinCancer' : int,
 'Sex_Female' : int,
 'Sex_Male' : int,
 'AgeCategory_18-24' : int,
 'AgeCategory_25-29' : int,
 'AgeCategory_30-34' : int,
 'AgeCategory_35-39' : int,
 'AgeCategory_40-44' : int,
 'AgeCategory_45-49' : int,
 'AgeCategory_50-54' : int,
 'AgeCategory_55-59' : int,
 'AgeCategory_60-64' : int,
 'AgeCategory_65-69' : int,
 'AgeCategory_70-74' : int,
 'AgeCategory_75-79' : int,
 'AgeCategory_80 or older' : int,
 'Race_American Indian/Alaskan Native' : int,
 'Race_Asian' : int,
 'Race_Black' : int,
 'Race_Hispanic' : int,
 'Race_Other' : int,
 'Race_White' : int})

    print(new_data1)

    prediction = heart_model.predict(new_data1)
    print(prediction)
    if prediction == 1:
        ans = 'You have heart disease'
    elif prediction == 0:
        ans = "You don't have heart disease"

    return render_template('index.html', prediction_ans = ans)


if __name__ == '__main__':
    app.run(debug=True)


