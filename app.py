
from flask import Flask, request, jsonify,app,url_for,render_template,redirect,flash
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
reg_model=pickle.load(open('model.pkl','rb'))# loading the model/deserializing the saved model
scaler=pickle.load(open('scaler.pkl','rb'))# loading the scaler for featue scaling

@app.route('/')
def home():
        return render_template('home.html')#it will render the home page


@app.route('/predict_api',methods=['POST'])# to test it from POSTMAN
def predict_api():
        data=request.json["data"] # it gets the values of the data key
        print(data)
        reshaped_data=np.array(list(data.values())).reshape(1,-1) # reshaping the data into 2d

        scaled_data=scaler.transform(reshaped_data)
        output=reg_model.predict(scaled_data)
        print(output[0])
        return jsonify(list(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
        data=[float(x) for x in request.form.values()]
        final_input=scaler.transform(np.array(data).reshape(1,-1))
        print(final_input)
        output=reg_model.predict(final_input)[0]
        print(output)
        return render_template('home.html',prediction_text='The predicted house price is {}'.format(output))



if __name__=='__main__':
        app.run(debug=True)
   

        




"""

pickled_model=pickle.load(open('model.pkl','rb'))

prediction=pickled_model.predict([[-0.41456707, -0.50512499, -0.58639669, -0.28154625, -0.75676687,
         1.19651116, -0.27871012,  0.5658082 , -0.862084  , -0.99251596,
        -0.21208981,  0.38165277, -1.18145831]])
print(prediction)
"""