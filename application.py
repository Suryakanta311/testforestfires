from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
#import ridge regreesor and standard scaler pickle
ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scaler=pickle.load(open('scaler.pkl','rb'))
@application.route("/")
def hello_world():
              return render_template('index.html')

@application.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    print("REQUEST METHOD:", request.method)

    if request.method == 'POST':

        print("POST DATA:", request.form)

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        print("INPUT VALUES RECEIVED")

        new_data = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        print("SCALED DATA:", new_data)

        result = ridge_model.predict(new_data)

        print("PREDICTION:", result)

        return render_template('home.html', results=result[0])

    return render_template('home.html', results="")

    return render_template('home.html')
if __name__=="__main__":
              application.run(host="0.0.0.0")      


