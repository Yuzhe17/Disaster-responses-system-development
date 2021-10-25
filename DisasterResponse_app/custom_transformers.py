from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

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


class extract_pos_num(BaseEstimator,TransformerMixin):
    def tokenize(self,text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens

    def count_pos_num(self,tokens):
        verb_num=0
        noun_num=0
        adj_num=0
        adv_num=0
        other_num=0
        pos_tags=nltk.pos_tag(tokens)

        for _,pos in pos_tags:
            if pos.startswith('J'):
                adj_num += 1
            elif pos.startswith('V'):
                verb_num += 1
            elif pos.startswith('N'):
                noun_num += 1
            elif pos.startswith('R'):
                adv_num += 1
            else:
                other_num += 1
        return dict(verb=verb_num, noun=noun_num, adj=adj_num, adv=adv_num, other=other_num)

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        pos_num = pd.Series(X).apply(self.tokenize).apply(self.count_pos_num).values
        return pd.DataFrame(list(pos_num))
