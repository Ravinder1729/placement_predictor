from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        ssc = float(request.form.get("ssc"))
        inter = float(request.form.get("inter"))
        mba = float(request.form.get("mba"))
        grad = float(request.form.get("grad"))
        exp = request.form.get("exp")
        spec = request.form.get("spec")
        
        
        label_encoder = LabelEncoder()
        exp = label_encoder.fit_transform([exp])[0] 
        spec = label_encoder.fit_transform([spec])[0] 
        
        data_point = np.array([ssc, inter, mba, grad, exp, spec]).reshape(1, -1)
        
        model = pickle.load(open(r"D:\innomatics\ML\mb2.pkl", 'rb'))
        prediction = model.predict(data_point)
        
        return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
