"""
This is the Entry point for Training the Machine Learning Model.



"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing

from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()
            data.drop('weather_description',inplace=True,axis=1)


            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)

            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)

            # data  = preprocessor.logTransformation(data)

            # encode the prediction data
            data_encoded = preprocessor.encodeCategoricalValues(data)
            ###Time features

            data = preprocessor.create_timefeatures(data_encoded)


            print(data_encoded)
            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name="traffic_volume")
            # drop the columns obtained above






            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=36)



            model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

            #getting the best model for each of the clusters
            best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

            #saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object,self.log_writer)
            save_model=file_op.save_model(best_model,best_model_name)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception