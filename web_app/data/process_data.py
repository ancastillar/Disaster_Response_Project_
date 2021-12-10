# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import unicodedata
import re
import string
import sys
from sklearn.feature_selection import VarianceThreshold


    
def load_data(messages_filepath, categories_filepath):

    """ This function help with load data
    Inputs:
            messages_filepath (String): file path for a messages dataframe
            categories_filepath (String): file path for a categories dataframe
    Outputs:
            df (Dataframe): DataFrame merge from messages and categories dataframes
    """
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on ="id", how= "inner")
    
    return df


def clean_data(df):

    """ 
    This function help with basic clean data
    Inputs:
            df (Dataframe)
    Outputs:
            df (Dataframe): Clean dataframe
    """
  
   
#-------------------------------------------------------------------------------
    
    def elimina_tildes(cadena):

        """ Drop accents for each string

        Inputs:
            cadena (String): Text data
        Outputs:
            cadena (String): Text data without accent
        """
        s = ''.join((c for c in unicodedata.normalize('NFD',cadena) if unicodedata.category(c) != 'Mn'))
        
        return s
    
    def schema_dataframe(df):
      
      """ 
      This function create the targets variables from categorie data
      Inputs:
            df (Dataframe): Dataframe with categories in one columna
      Outputs:

            df (Dataframe): Dataframe with one column per categorie
      
      """

      categories = df.categories.str.split(pat= ";", expand = True)
      row = categories.iloc[:1,:]

      category_colnames = list(row.values[0])


      for i,cat in enumerate(category_colnames):
      
          cat = cat[:-2]
          category_colnames[i] = cat
          
      categories.columns = category_colnames
      
      for column in categories:
  
          categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)
      
      df = df.drop(["categories", "original"], axis=1)
      
      df = pd.concat([df, categories], axis=1)
      
      df.drop_duplicates(inplace= True)

      return df
  

#-------------------------------------------------------------------------------

    def clean_text_p(text):
        
        """ remove text in square brackets, remove punctuation and remove words containing numbers.
        Inputs:
              text (String): String data
        Output:
              text (String): Clean text data
        """
        
        text = re.sub('\[.*?\!]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[‘’“”?¿…,#&¡]', '', text)
        text = re.sub('\n', '', text)
        text= re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text)

        return text

#-------------------------------------------------------------------------------

    def data_consistency(df):

          """
          This function clean a Dataframe form incongruence data and variables
          
          Inputs:
                    df: Dataframe 
          Outputs:  
                    df: Dataframe without incongruence columns

          """
          est_ = pd.DataFrame(df.loc[:, "related":].describe().loc["max",:]).T

          for  row in est_.itertuples(index=False, name=None):
            for i,c in enumerate(row):  
              if (c > 1):
              
                df  = df[df[est_.columns.values[i]]<=1]
              
              if (c==0):
              
                df = df.drop([est_.columns[i]], axis=1)

          return df

#------------------------------ RUN --------------------------------------------
    threshold_n = 0.9999
    
    df = schema_dataframe(df)
   

    var_text = [ "message", "genre"]
    for var in var_text:

        df[var] = df[var].apply(lambda x: x.lower())
        df[var] = df[var].apply(lambda x: x.strip())
        df[var] = df[var].apply(lambda x: elimina_tildes(x))
        df[var] = df[var].apply(lambda s: clean_text_p(s) if type(s) ==str else s )

    df = data_consistency(df)
    
    sel = VarianceThreshold(threshold=(threshold_n* (1 - threshold_n) ))
    num = df.select_dtypes(exclude="object")   
    sel_var=sel.fit_transform(num)
    select_num_cols = list(num[num.columns[sel.get_support(indices=False)]].columns)

    select_num_cols = ["message", "genre"] +select_num_cols
    select_num_cols.remove("id")
    
    df = df[select_num_cols]
 
    return df
    


def save_data(df, database_filename, data_table_name = "data_project"):

    """
          This function save the preprocess dataframe
          
          Inputs:
                    df: Dataframe
                    database_filename (String): File name for database where we will save data 
                    data_table_name (String): String with the table name
          Outputs:  
                    None

          """
    engine = create_engine('sqlite:///'+database_filename )
    df.to_sql(data_table_name, engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)
       
        print('Cleaning data...')
        
        df = clean_data(df)
    
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    
    main()