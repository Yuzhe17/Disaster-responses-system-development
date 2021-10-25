import pandas as pd
import os
import re
import sys
import logging
import datetime
import sqlite3
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import pickle


logging.basicConfig(filename='disaster_response.log', level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
logging.info('start running: {}'.format(str(datetime.datetime.now())))

def get_wordnet_pos(treebank_tag):
    '''map treebank tag to wordnet pos

    args:
        treebank_tag: str. treebank tag from nltk.pos

    '''

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def data_loading(db_path):
    '''loading data from database

    args:
        None

    returns:
        X: numpy array of disaster response text
        y: numpy array containing multilabels
    '''

    con=sqlite3.connect(db_path)
    cur=con.cursor()
    df_disaster=pd.read_sql("SELECT * FROM disaster_response",con)
    # X=df_disaster['message'].values
    # y=df_disaster.iloc[:,6:].values

    return df_disaster

def tokenize(text):
    '''tokenize text 
    
    args:
        text: str. text file to be tokenized

    returns:
        tokens_list: list. A list of tokens.
    '''

    text=re.sub(r'[^\w\s]','',text).lower()#Normalization
    tokens_pos=pos_tag(word_tokenize(text))#tokenization and pos tagging
    tokens_list=[]
    wnl=WordNetLemmatizer()
    

    for word,pos in tokens_pos:
        tokens_list.append(wnl.lemmatize(word,get_wordnet_pos(pos)) if get_wordnet_pos(pos) else word)
    
    return tokens_list

class extract_starting_verb(BaseEstimator,TransformerMixin):
    '''Custom transformer that determine whether the first word is verb
    '''
    def is_starting_verb(self,text):
        '''determine if the first word in a sentence is a verb

        args:
            text(str): a text containing sentences

        returns:
            bool: the return value. True when the first word in a sentence is verb, False otherwise.
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_tagged = pd.Series(X).apply(self.is_starting_verb)
        return pd.DataFrame(X_tagged)

def create_model():
    '''instantiate and train a machine learning model
    args:
        df: a pandas dataframe. A dataframe containing data and corresponding label.

    returns:
        classifier: pickle file. A machine learning model for classification.
    '''
    tfidf=TfidfVectorizer(tokenizer=tokenize)
    starting_verb_extractor = extract_starting_verb() 

    classifier=AdaBoostClassifier() #AdaBoostClassifier is selected after comparing the perfermances of different classifiers

    pipe=Pipeline([('feature',FeatureUnion([('densetfidf',tfidf),('starting_verb',\
        starting_verb_extractor)])),('classifier',MultiOutputClassifier(classifier))])

    return pipe

def save_model(model,classifier_path):
    '''save the model as pickle file

    args:
        model: sklearn predictor

    returns:
        classifier_path: path where a trained classifier is stored
    '''
    with open(classifier_path,"wb") as f:
        pickle.dump(model,f)

def optimize_model(classifier, X_train,y_train):
    '''tune hyperparameters of the classifier

    args:
        classifier: sklearn predictor. 
        X_train(ndarray): training array of the training set
        y_train(ndarray): target array of the training set

    returns:
        sklearn predictor

    '''

    gridcv=GridSearchCV(classifier,param_grid={'feature__densetfidf__min_df':[1,15],'feature__densetfidf__ngram_range':[(1,1),(1,2)],'classifier__estimator__n_estimators':[300,400]},\
        verbose=1,scoring='f1_weighted')
    gridcv.fit(X_train.flatten(),y_train)

    return gridcv.best_estimator_

def evaluate_model(classifier, X_test, y_test):
    '''evaluate the model

    args:
        classifier: sklearn predictor. 
        X_train(ndarray): testing array of the testing dataset
        y_train(ndarray): target array of the testing dataset

    returns:
        f1: f1 score
    '''

    y_pre=classifier.predict(X_test.flatten())
    f1=f1_score(y_test,y_pre,average='weighted')

    return f1
    

def main():
    '''create, train and a classification model
    '''
    df_disaster=data_loading(db_path)
    df_disaster=df_disaster.drop(columns=['child_alone'])
    X=df_disaster['message'].values.reshape(-1,1)
    y=df_disaster.iloc[:,6:].values
    
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
    
    #create a classifier
    classifier=create_model()

    #tune hyperparameters of the classifer
    classifier=optimize_model(classifier,X_train,y_train)

    #evaluate the classifier
    f1=evaluate_model(classifier,X_test,y_test)
    logging.info('f1 score {}, cv results {}'.format(f1, classifier.cv_results_))

    #save the model
    save_model(classifier,classifier_path)
        

if __name__=='__main__':
    db_path,classifier_path = sys.argv[1:]
    main()








    

    





