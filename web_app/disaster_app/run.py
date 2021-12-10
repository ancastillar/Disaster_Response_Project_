from disaster_app import app
import json
import plotly
import pandas as pd
import sqlite3
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from models.transformers_class import TokenizerTransformer, PadSequencesTransformer, read_glove_vecs
from sqlalchemy import create_engine
from os.path import isfile, join
from os import listdir
from sklearn.pipeline import Pipeline
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import os

# load data
conn = sqlite3.connect("data/DisasterResponse.db")

df = pd.read_sql('SELECT * FROM data_project', conn)
### direct

def plot_words(df, id=False):
    
    if id==True:
        for i in ["direct", "news", "social"]:
            wordcloud = WordCloud(max_words = 500 , background_color="white",width = 1600 , height = 800, stopwords =["please", "rt" "as","on", "is","are","of","this", "these", "that","from", "in", "with","wich","an","but", "by","were", "on","have", "been","would", "like", "was", "we", "they", "you", "or", "do", "dont", "people", "thank", "us", "my","im","said", "will", "to", "you","the", "and", "it", "http", "for", "are", "was", "me", "because","am", "be", "who", "what", "hello", "there", "not", "at", "if", "which", "ha", "as", "also", "has", "their", "its", "had", "all"]).generate(" ".join(df[df.genre==i]["message"]))
            wordcloud.to_file("app/img/"+i+"_msn.png")

plot_words(df, id =False)            
            
            
my_tokenizer = TokenizerTransformer()
word_to_vec_map = read_glove_vecs("models/glove.6B.50d.txt")
my_padder = PadSequencesTransformer( word_to_vec_map=word_to_vec_map)


preprocess_pipeline =  Pipeline([('tokenizer', my_tokenizer), ('padder', my_padder)])



# load model
models_names = os.listdir("models/models_clfs/")
models = {}
for name in models_names:
    models[name[:-6]] = joblib.load("models/models_clfs/"+name)
    del models[name[:-6]]




# index webpage displays cool visuals and receives user input text for model

@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = [request.args.get('query', '') ]
    
    query_df = pd.DataFrame(query, columns=["message"])

    # use model to predict classification for query
    query_preprocess = preprocess_pipeline.fit_transform(query_df)
 
    
    classification_results = {}
    for name, model in models.items():
        classification_results[name] = model.predict([query_preprocess.flatten() ])
     

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
    
    
