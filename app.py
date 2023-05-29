from flask import Flask
import requests
from flask import render_template
import pickle
import numpy as np 
import os 

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#set file directory path

#MODEL_PATH = os.path.join(APP_ROOT, './model.pkl')
# set path to the model

model = pickle.load(open('/python-docker/models/model.pkl', 'rb'))
# load the pickled model

@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])

def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction  = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text='Perksovite formation should be {}'.form(output))

@app.route('/predict_api', methods=['POST'])

def predict_api():

    data = request.get.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)