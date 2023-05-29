import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/api', methods=['POST'])
def predict():

    #Get teh data from the Post request
    data = request.get_jon(force=True)

    #Make Prediction using model loaded from disk as per the data

    prediction = model.predict([np.array[data['exp']]]) 

    #Take the first value of prediction

    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexectedly. Please contact Joshua Fordyce")

