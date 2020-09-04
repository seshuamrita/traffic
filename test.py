from wsgiref import simple_server
from flask import Flask, request
from flask import Response
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

import pandas as pd
import shutil
app = Flask(__name__)
@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    if request.method == 'POST':
        'folderpath'='seshu'
        path = request.form['folderpath']
        return path




if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
	#app.run(debug=True) # running the app