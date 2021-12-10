# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from transformers_class import TokenizerTransformer, PadSequencesTransformer,  read_glove_vecs
from sklearn.metrics import classification_report

#imbalanced class
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
#model

from catboost import CatBoostClassifier
###save models
import joblib




#--------------------------------------------------------------------------------------

def load_data(database_filepath):
    
    """ This function help with the connection with database and load the data
    Inputs:
        database_filepath (String): filepath with the location of database
    Outputs:
         df (Dataframe):  datframe with the data
    """
    conn = sqlite3.connect(database_filepath)

    # run a query
    df = pd.read_sql('SELECT * FROM data_project', conn)
    
    return df



#---------------------------------------------------------------------------------------------

def preprocess_data(df, word_to_vec_map):
    
    """ This function preprocess data for modeling: Tokenization, sequence and create embedding
    Inputs:
            df (dataframe): Dataframe with text and target variables
            word_to_vec_map (Dictionary): Dictionary with embedding
    Outputs:
            X_train (Dictionary): Dictionary with array train data for each model
            X_test (Dictionary): Dictionary with array test data for each model
            X_val (Dictionary): Dictionary with array validate data for each model
            target_model_train (Dictionary): Dictionary with array train target data for each model
            target_model_test (Dictionary): Dictionary with array test target data for each model
            target_model_val (Dictionary): Dictionary with array validate target data for each model
    """
    my_tokenizer = TokenizerTransformer()
    my_padder = PadSequencesTransformer( word_to_vec_map=word_to_vec_map)


    preprocess_pipe =  Pipeline([
        ('tokenizer', my_tokenizer),         
        ('padder', my_padder)])


    X = df[["message"]]

    y = df.loc[:,"related":]
    


    X = preprocess_pipe.fit_transform(X, y)
    
    #joblib.dump(preprocess_pipe, "models/preprocess_pipeline.pkl")

    X_train = {}

    X_test = {}
    
    X_val = {}

    target_model_train = {}


    target_model_test = {}
    
    target_model_val = {}


    for col in y.columns:

      X_train[col], X_test[col], target_model_train[col], target_model_test[col] = train_test_split(X,y[col], test_size = 0.2,
                                             random_state=7, shuffle= True, stratify=y[col] ) 
      
      X_test[col], X_val[col], target_model_test[col], target_model_val[col] = train_test_split(X_test[col],target_model_test[col],                                                                                                 test_size=0.1, random_state=7,                                                                                           shuffle=True, stratify=target_model_test[col]) 
    print(X_train["related"].shape)
    return X_train, X_test, X_val, target_model_train, target_model_test, target_model_val

#--------------------------------------------------------------------------------------


def build_model(X_train, X_val, target_model_train, target_model_val):
    
        """ This function preprocess data for modeling: Tokenization, sequence and create embedding
        Inputs:
                X_train (Dic): Dictionary with data for training model 
                X_val (Dic): Dictionary with data for validate model
                target_model_train (Dic): Dictionary with target for training
                target_model_test (Dic): Dictionary with target for validate
        Outputs:
                models (Dic): Dictionary with all the models

        """

        models = {}
        
        params_grid ={
                      'model__depth': [4,6],
                      'model__n_estimators': [75, 250, 300], #number of trees
                      'model__l2_leaf_reg' : [3, 5, 7]}



        for model, target in target_model_train.items():

          print(model)

          w = len(target[target==0]) / len(target[target==0])
          my_clf = CatBoostClassifier(loss_function='Logloss',random_state=42, scale_pos_weight= w, n_estimators = 250,                                                  l2_leaf_reg=3,eval_metric="AUC" )

          my_smote = oversample = SMOTE()

          pipeline_nlp = Pipeline([
              ('smote', my_smote),
              ('model', my_clf)
              ])

          #If you have time and enough resources please uncomment
            
          #kfolds = StratifiedKFold(3)

          #grid_pipeline_nlp = RandomizedSearchCV(pipeline_nlp, param_distributions = params_grid, n_jobs=-1, cv=kfolds.split(X_train[model] ,                                    target_model_train[model]),scoring='accuracy')

          print("fitting...")
          
          pipeline_nlp.fit( X_train[model], target_model_train[model], model__eval_set = [(X_val[model], target_model_val[model])], 
                           model__verbose=True, model__early_stopping_rounds=15)

          models[model] = pipeline_nlp
            
          # or...
   
          #grid_pipeline_nlp.fit( X_train[model], target_model_train[model], model__eval_set = [(X_val[model], target_model_val[model])], 
          #                 model__verbose=True, model__early_stopping_rounds=15)
          #best_model = grid_pipeline_nlp.best_estimator_ 
          #models[model] = best_model
        
        return models

#-----------------------------------------------------------------------------------------------------------


def evaluate_model(models, X_test, y_test): 
    
   """ This function evaluate the results of our model 
   
   Inputs:
         models (Dictionary): Dic with all Classifier model
         X_test (Dictionary): Dic with data for prediction
         y_test (Array): Dic with True labels
         
   Outputs:
   
          None
   """
   y_pred = {}
    
   for model, data_test in X_test.items():
        
        print("Evaluating...",model)
        
        y_pred = models[model].predict(data_test)
        
        print(classification_report(y_test[model], y_pred))
         
#------------------------------------------------------------------------------------

def save_model(models, model_filepath): 
    
          for name_model, model_ in models.items():
            
              joblib.dump(model_, model_filepath + name_model+ "_"+ ".hdf5")

def main():
    
    if len(sys.argv) == 4:
        
        database_filepath, model_filepath, glove_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print('Loading embedding...\n    EMBEDDING: {}'.format(glove_filepath))
        
        df = load_data(database_filepath)
        
        print(df)
        
        word_to_vec = read_glove_vecs(glove_filepath) 

        print('Preprocessing data...')
        
        X_train, X_test, X_val, target_model_train, target_model_test, target_model_val = preprocess_data(df,                                                                                                                                                                                                                                         word_to_vec_map=word_to_vec)
        
        print('Building model and training models...')
        models = build_model(X_train, X_val, target_model_train, target_model_val)
        print(models.keys())

        print('Evaluating model...')
        
        evaluate_model(models, X_test, target_model_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        
        save_model(models, model_filepath)

        print('Trained model saved!')

    else:
        
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()