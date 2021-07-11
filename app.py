
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

model = pickle.load(open('lin_reg.pkl', 'rb'))
min_max_sc = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        AdultMor = float(request.form['adult_mor'])
        Income= float(request.form['income'])       
        HIV=float(request.form['hiv'])
        BMI=float(request.form['bmi'])
        Per_expenditure=float(request.form['per_exp'])
        Thinness5_9=float(request.form['thinness5_9'])
        GDP =float(request.form['gdp'])
        school=float(request.form['school'])
        Diphtheria=float(request.form['dipth'])
        polio=float(request.form['polio'])
        under_5=float(request.form['under_5'])
        thinness1_19=float(request.form['thinness1_19'])
        X_scaled = min_max_sc.transform([[AdultMor,Income,HIV,BMI,Thinness5_9,Per_expenditure,thinness1_19,polio,GDP,under_5,school,Diphtheria]])
              
        prediction=model.predict(X_scaled)
                
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_text="Sorry please re-enter the details")
        else:
            return render_template('index.html',prediction_text="The expected life in years is {} ".format(output))
    else:
        return render_template('index.html')
        
 
if __name__=="__main__":
    app.run(debug=True)       
