import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation



class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()



            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            #data  = preprocessor.logTransformation(data)


            #encode the prediction data
            data_scaled = preprocessor.encodeCategoricalValuesPrediction(data)
            ###Time features
            data=preprocessor.create_timefeatures(data)

            #data=data.to_numpy()
            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)



            model = file_loader.load_model('XGBOOST')
            result.model.predict(data)

            result = pandas.DataFrame(result,columns=['Predictions'])
            result['Item_Identifier']  = data["Item_Identifier"]
            result["Outlet_Identifier"] = data["Outlet_Identifier"]
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path

    def prediction_from_user(self):
        try:
            dbConn = pymongo.MongoClient("mongodb://localhost:27017/")  # opening a connection to Mongo
            db = dbConn['HOURLYTRAFFICDETAILS']  # connecting to the database called DB
            # reading the inputs given by the user
            holiday = request.form['holiday']
            if (holiday == 'None'):
                Holiday_True = 0
            else:
                Holiday_True = 1
            temp = float(request.form['temp'])  # float variable
            # rain_1h = float(request.form['rain'])  #float variable
            # snow_1h = float(request.form['snow'])  #float variable
            clouds_all = int(request.form['cloud'])  # float variable
            weather_main = request.form['weather']  # Categorical variable will be encoded
            date_time = request.form['Date_time']
            previous_hour = request.form['previoushourtraffic']
            list = ['CLOUDS', 'MIST', 'RAIN', 'SNOW', 'OTHERS']
            # Creating a dictionary
            dict_pred = {'holiday': Holiday_True, 'temp': temp, 'rain_1h': rain_1h, 'snow_1h': snow_1h,
                         'clouds_all': clouds_all,
                         'date_time': date_time, 'weather_main': weather_main, 'previous_ihr': previous_hour
                         }

            # Creating a dataframe
            df = pd.DataFrame(dict_pred, index=[0, ])
            # Converting to DateTime
            df['date_time'] = pd.to_datetime(df.date_time)
            df['weekday'] = df.date_time.dt.weekday  # Monday is 0 and Sunday is 6
            df['hour'] = df.date_time.dt.hour
            df['month'] = df.date_time.dt.month
            df['year'] = df.date_time.dt.year
            df['weather_main'] = np.where(df['weather_main'].upper().isin(list), df['weather_main'], 'OTHERS')

            # Applying one-hot encoding
            filename = 'OHE.pkl'
            # Opening and loading the pickle file
            ohe = pickle.load(open(filename, 'rb'))
            weather_main_df = pd.DataFrame(ohe.transform(df[['weather_main']]).toarray())
            weather_main_df.columns = ohe.get_feature_names(['weather'])
            # Join the encoded dataset with the given one
            df = df.join(weather_main_df)
            # Dropping categorical column
            df.drop('weather_main', axis=1, inplace=True)
            # setting the index to date_time value
            df.set_index('date_time', inplace=True)
            # print df info
            # print(df.head())
            model = file_loader.load_model(xgb)

            # Prediction using the loaded pickle file
            predictionout = model.predict(df)
            table = db['TRAFFIC']
            mydict = {'holiday': Holiday_True, 'temp': temp, 'rain_1h': rain_1h, 'snow_1h': snow_1h,
                      'clouds_all': clouds_all,
                      'date_time': str(date_time), 'weather_main': weather_main, 'traffic_volume': prediction,
                      'previous_1hr': previous_hour
                      }  # saving that detail to a dictionary
            x = table.insert_one(mydict)
            return predictionout
        except Exception as e:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % e)
            raise e
