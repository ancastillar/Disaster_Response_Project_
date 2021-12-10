import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import nltk 
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from collections import Counter
from tqdm import tqdm
import unicodedata
import re
import string
from string import digits

#-----------------------------------------------------------------
#Definition of stop words

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add("people")
STOPWORDS.add("thank")
STOPWORDS.add("thanks")
STOPWORDS.add("please")
STOPWORDS.add("santiago")
STOPWORDS.add('http')





            
def read_glove_vecs(glove_file):

            """
            This fuction read the embedding

            Inputs: 
                  Glove_file (String): Path of glove file

            Outputs:

                word_to_vec_map: word to 50d vector mapping output

            """

            with open(glove_file, 'r') as f:

                words = set()
                word_to_vec_map = {}

                for line in f:
                    line = line.strip().split()
                    curr_word = line[0]
                    words.add(curr_word)
                    word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            print("Here")

            return word_to_vec_map

#----------------------------------------------------------------------------------

class TokenizerTransformer(BaseEstimator, TransformerMixin):
    """
    This class clean and preprocess text data

    Inputs: 
          num_rare_words (Integer): Number of rare words that you want to drop
          X: Dataframe with the text data

    Outputs:

        X: Dataframe with text data without stop words, emojis...

    """

    def __init__(self, num_rare_words=15 ):

        self.num_rare_words =  num_rare_words
     
        

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
      
          
       def rarewords_text(X):

              """ This function extract the rare words from text data
              Inputs:

                    X: DataFrame
              Outputs:
                    rare_w (list): List with n rare words
              """

              cnt = Counter()
              for text in X.values:

                  for word in text.split():

                      cnt[word] += 1
                      
              n_rare_words = self.num_rare_words
              rare_w = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

              return rare_w

       RAREWORDS = rarewords_text(X.message)
     
       def clean_sentences(text, rarewords):

              """ This function clean the data text: removing rare words, stop words, short words...
              Inputs:

                    X: DataFrame
                    rarewords (list): List of rarewords to remove

              Outputs:
                    X (Dataframe): Dataframe with text data clean.
              """
              
              remove_digits = str.maketrans('', '', digits)
              text = text.translate(remove_digits)    
              text = RegexpTokenizer(r'\w+').tokenize(text)
              text = [s.lower() for s in text]
              text = [s for s in text if s not in set(stopwords.words('english'))]
              text = [s for s in text if len(s)>3]
              text = ' '.join(text)
              text = " ".join([word for word in str(text).split() if word not in rarewords])
              text = " ".join([WordNetLemmatizer().lemmatize(word) for word in text.split()])

              return text 


             
       X["text_clean"] = X["message"].apply(lambda text: clean_sentences(text, rarewords= RAREWORDS))
              
       X_transformed = X.text_clean

       return  X_transformed 
    
#--------------------------------------------------------------------------------------

class PadSequencesTransformer(BaseEstimator, TransformerMixin):

    """
    This class create the embedding for train the model

    Inputs: 
          word_to_vec_map (Dictionary): Dic with the words embedding in this case Glove 50D.
          X: Dataframe with the text data

    Outputs:

          X (Array): Array with vector of text data

    """

    def __init__(self, word_to_vec_map):

        self.word_to_vec_map = word_to_vec_map

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
       

        def prepare_sequence(X, word_to_vec_map):

            """
            This function create a vector with the text data

            Inputs: 
                  word_to_vec_map (Dictionary): Dictionary with the word embedding
                  X: Dataframe with the text data

            Outputs:

                X (Array): Array with vector of text data

            """
            traintest_X = []

            for sentence in tqdm(X.values):

                sequence_words = np.zeros((word_to_vec_map['cucumber'].shape))

                for word in sentence.split():

                    if word in word_to_vec_map.keys():

                        temp_X = word_to_vec_map[word]
                    else:
                        temp_X = word_to_vec_map['#']

                    sequence_words+=(temp_X)/len(sentence)

                traintest_X.append(sequence_words)

            return np.array(traintest_X)

        X_padded = prepare_sequence(X, self.word_to_vec_map)

        return X_padded
