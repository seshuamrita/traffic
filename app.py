from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import pickle
import sklearn
import pymongo
import os
from validation import (validate_int, validate_float,validate_structure,validate_text)
import flask_monitoringdashboard as dashboard
from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import json

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__) # requirement for AWS deployment

dashboard.bind(app)
CORS(app)

#model = pickle.load(open("xgboost_model.pickle", "rb"))
model = pickle.load(open('random_forest.pickle', 'rb'))

@app.route('/', methods = ['GET']) # route to display home page
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/download', methods=['GET', 'POST'])
def download_file():
    path = "test1.csv"
    return send_file(path, as_attachment=True)

@app.route("/predict", methods = ['POST']) # route to show the single prediction in a Web UI
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            dbConn = pymongo.MongoClient("mongodb://localhost:27017/")  # opening a connection to Mongo
            db = dbConn['HOURLYTRAFFICDETAILS']  # connecting to the database called DB
            #reading the inputs given by the user
            holiday = request.form['holiday']
            if (holiday == 'None'):
                Holiday_True = 0
            else:
                Holiday_True = 1
            temp = float(request.form['temp']) #float variable
            rain_1h = float(request.form['rain'])  #float variable
            snow_1h = float(request.form['snow'])  #float variable
            clouds_all = int(request.form['cloud']) #float variable
            weather_main = request.form['weather'] #Categorical variable will be encoded
            date_time = request.form['Date_time']

            #Creating a dictionary
            dict_pred = {'holiday': Holiday_True,'temp': temp,'rain_1h': rain_1h,'snow_1h': snow_1h,'clouds_all': clouds_all,
                        'date_time' : date_time,'weather_main':weather_main
                         }

            #Creating a dataframe
            df = pd.DataFrame(dict_pred,index=[0, ])
            #Converting to DateTime
            df['date_time'] = pd.to_datetime(df.date_time)
            df['weekday'] = df.date_time.dt.weekday #Monday is 0 and Sunday is 6
            df['hour'] = df.date_time.dt.hour
            df['month'] = df.date_time.dt.month
            df['year'] = df.date_time.dt.year

            #Applying one-hot encoding
            filename = 'onehotencoder.pickle'
            #Opening and loading the pickle file
            ohe = pickle.load(open(filename,'rb'))
            weather_main_df = pd.DataFrame(ohe.transform(df[['weather_main']]).toarray())
            weather_main_df.columns = ohe.get_feature_names(['weather'])
            #Join the encoded dataset with the given one
            df = df.join(weather_main_df)
            #Dropping categorical column
            df.drop('weather_main',axis=1,inplace=True)
            #setting the index to date_time value
            df.set_index('date_time',inplace=True)
            #print(df.head())

            #Prediction using the loaded pickle file
            prediction = model.predict(df)
            #adding into DB
            table = db['TRAFFIC']
            mydict = {'holiday': Holiday_True, 'temp': temp, 'rain_1h': rain_1h, 'snow_1h': snow_1h,
                      'clouds_all': clouds_all,
                      'date_time': date_time, 'weather_main': weather_main, 'traffic_volume': str(prediction)
                      }  # saving that detail to a dictionary
            x = table.insert_one(mydict)

            #printing into the console
            print("Prediction is ", np.rint(prediction))
            #Showing the result in UI
            return render_template('results.html',prediction=np.rint(prediction[0]))
            #return str(prediction[0])
        except Exception as e:
            print("The Exception message is:", e)
            return jsonify("error:Something is wrong")
    else:
        return render_template('index.html')


@app.route("/predict_file", methods = ['POST'])
@cross_origin()
def predict_file():
    try:
        # reading the given csv file
        df_test = pd.read_csv(request.files['file'])
        # performing some basic data validation
        validation = Validation()
        validity = validation.method_validation(df_test)

        if type(validity) == str:
            return render_template('resultsBulk.html', e='Invalid data format')

        df_test['date_time'] = pd.to_datetime(df_test.date_time)
        df_test['weekday'] = df_test.date_time.dt.weekday  # Monday is 0 and Sunday is 6
        df_test['hour'] = df_test.date_time.dt.hour
        df_test['month'] = df_test.date_time.dt.month
        df_test['year'] = df_test.date_time.dt.year

        # Applying one-hot encoding
        filename = 'onehotencoder.pickle'
        # Opening and loading the pickle file
        ohe = pickle.load(open(filename, 'rb'))
        weather_main_df = pd.DataFrame(ohe.transform(df_test[['weather_main']]).toarray())
        weather_main_df.columns = ohe.get_feature_names(['weather'])
        # Join the encoded dataset with the given one
        df_test = df_test.join(weather_main_df)
        # Dropping categorical column
        df_test.drop('weather_main', axis=1, inplace=True)
        # setting the index to date_time value
        df_test.set_index('date_time', inplace=True)

        #Prediction using the loaded pickle file
        prediction = model.predict(df_test)
        #printing into the console
        print("Prediction is ", str(list(prediction)))
        result = {}
        for i in range(0, df_test.shape[0]):
            result[i] = np.rint(prediction[i])
        #Showing the result in UI
        return render_template('resultsBulk.html', r=result)
        #return str(list(prediction))
        #return jsonify(str(result))

    except Exception as e:
        print("The Exception message is:", e)
        return jsonify("error:Something is wrong")



@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:

        path = r'Training_Batch_Files'




        train_valObj = train_validation(path) #object initialization

        train_valObj.train_validation()#calling the training_validation function


        trainModelObj = trainModel() #object initialization
        trainModelObj.trainingModel() #training the model for the files in the table



    except ValueError:


        return Response("Error Occurred! %s" % ValueError)

    except KeyError:


        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
	#app.run(debug=True) # running the app