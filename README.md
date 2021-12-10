# Disaster Response Pipeline Project

In this project, an application was built to classify messages given by the user in one of the 36 categories. The algorithms used are Catboots, the metric used to evaluate each model was the AUC. On the other hand, it was found that some of the classes were too unbalanced, so to avoid overfitting in these algorithms, balancing techniques were used, in this case SMOTE.

## Link to Webapp http://disasterprojectappnatalia.herokuapp.com/
![plot](./images_web_app/app_1.PNG)
![plot](./images_web_app/app_2.PNG)


## About this repository
In this repository you will find:

* web_app: This file contains all the structure for clean and preprocess data, modelling and desing app.
* web_app/models: In this file you find a glove embedding, train_classifier.py and transformers_class (this class help us with preprocess data). All the models are save in models_clfs.
* web_app/data: Here you can find the datasets and the database create for cleaning data. The clean has done with process_data.py.
* web_app/app: Here you can find all scripts for desing application.

## Model and results

Thirty-six classifiers were developed, using the Catboost algorithm. The lower metric was 0.70 
 and the upper metric was 0.93 AUC. You can check the results running the models (Please see the instructions).

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/models_clfs/ models/glove.6B.50d.txt`
 Note: You need stay into web_app file for run the above commands
 
2. Run the following command in the app's directory to run your web app.
    `python app/run.py` or python/disaster_response_app.py

3. Go to http://0.0.0.0:3001/


## Prerequisites

To install the flask app, you need:

* python3
* python packages in the requirements.txt file

Install the packages with

* pip install -r requirements.txt
* Please uncompress the file with glove embedding to running the models
## Acknowledgements
Thanks to Davivienda to support me through this course
