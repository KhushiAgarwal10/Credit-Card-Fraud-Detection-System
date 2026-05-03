from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
import pandas as pd

xgb=pickle.load(open('modules/xgb.pkl','rb'))
scale=pickle.load(open('modules/scale.pkl','rb'))
sample_df= pickle.load(open('modules/sample.pkl','rb'))
data_with_class=pickle.load(open('modules/data_with_class.pkl','rb'))
data=pickle.load(open('modules/data.pkl','rb'))

  

app=Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')



@app.route('/predict', methods=['POST'])
def predict():
    
    amount = float(request.form['Amount'] or 0)
    sample = data_with_class.sample(1).iloc[0]

    if 'Class' in sample.index:
        sample = sample.drop('Class')

    features = sample.tolist()
    amount = np.log1p(amount)

    amount = scale.transform([[amount]])[0][0]

    
    features[-1] = amount

    input_array = np.array(features).reshape(1, -1)

    prob = xgb.predict_proba(input_array)[0][1]

    prediction = 1 if prob > 0.8 else 0

    result = "Fraud" if prediction == 1 else "Not Fraud"

    return render_template(
        'predict.html',
        prediction_text=result,
        probability=round(prob * 100, 2)
    )


if __name__=="__main__":
  app.run(debug=True)