import os
import sys
from flask import Flask,render_template,request
import requests

from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
        age=int(request.form.get('age')),
        workclass=request.form.get('workclass'),
        final_weight=int(request.form.get('final_weight')),
        education=request.form.get('education'),
        education_num=int(request.form.get('education_num')),
        marital_status=request.form.get('marital_status'),
        occupation=request.form.get('occupation'),
        relationship=request.form.get('relationship'),
        race=request.form.get('race'),
        sex=request.form.get('sex'),
        capital_gain=int(request.form.get('capital_gain')),
        capital_loss=int(request.form.get('capital_loss')),
        hours_per_week=int(request.form.get('hours_per_week')),
        native_country=request.form.get('native_country'))


        data = data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline()
        pred_value = predict_pipeline.predict(data)

        result = pred_value[0]
        return render_template('form.html',final_result = result)




if __name__ == '__main__':
    app.run(host='0.0.0.0')


