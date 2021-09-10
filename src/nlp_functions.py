# Amruta OXAI NLP_V3.1
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha

'''
NLP Processing functions and SKLearn Pipeline functions
'''
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn2pmml.feature_extraction.text import Splitter

import spacy
from spacy.lang.en import English
import plotly.express as px
from nltk import word_tokenize, FreqDist, RegexpTokenizer, pos_tag
from nltk.stem.porter import PorterStemmer
import string
import textblob
import nltk

# nltk.download('averaged_perceptron_tagger')

@st.cache(allow_output_mutation=True)
def count_vect(tag_pos):
    print('vectorizing...')
    if tag_pos:
        vect = CountVectorizer(analyzer='word', tokenizer=tokenize_pos)
    else:
        vect = CountVectorizer(analyzer='word', tokenizer=Splitter())
    #vect.fit(df[text_column])
    #vect_df = vect.transform(df[text_column])
    return vect

@st.cache(allow_output_mutation=True)
def tfidf(level, tag_pos):
    if tag_pos:
        tokenizer = tokenize_pos
    else:
        tokenizer = None
    print('vectorizing...')
    if level == 'Word':
        vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, tokenizer=tokenizer)
    elif level == 'N-Gram':
        vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, tokenizer=tokenizer)
    elif level == 'Character':
        vect = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000, tokenizer=tokenizer)
    #vect.fit(df[text_column])
    #tfidf_df = vect.transform(df[text_column])
    return vect

def tokenize_pos(tokens):
    """Add POS-tags to each token."""
    return [token+"_POS-"+tag for token, tag in pos_tag(tokens)]

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        # Doc.vector defaults to an average of the token vectors.
        # https://spacy.io/api/doc#vector
        return [self.nlp(text).vector for text in X]

@st.cache
def stemmer(text_list):
    print('stemming...')
    stm = PorterStemmer()
    return_list = []
    for i in range(len(text_list)):
        return_list.append(stm.stem(text_list[i]))

    return return_list

#@st.cache
def spacy_lemmatize(text_list):
    print('lemmatizing...')
    lemma_list = []
    for token in text_list:
        if token.is_stop is False:
            lemma_list.append(token.lemma_)

    return(lemma_list)

@st.cache
def remove_stopwords(text_list):
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    return_list = []
    for i in range(len(text_list)):
        if text_list[i] not in spacy_stopwords:
            return_list.append(text_list[i])
    return return_list

@st.cache
def count_characters(df, text_column):
    char_count = df[text_column].apply(len)
    return char_count

@st.cache
def plot_char_count(df, label_column):
    fig = px.box(df, y='char_count', title='Character Count per Record',
                 color=label_column)
    return fig

@st.cache
def lowercase(text):
    return text.lower()

#@st.cache
def tokenize(text):
    '''lower case, remove puncuation, tokenize
    '''
    print('tokenizing..')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens

@st.cache
def detokenize(text_list):
    print('detokenizing...')

    return ' '.join(text_list)

@st.cache
def count_words(df, text_column):
    word_count = df[text_column].apply(tokenize).apply(len)
    return word_count

@st.cache
def plot_word_count(df, label_column):
    fig = px.box(df, y='word_count', title='Word Count per Record',
                 color = label_column, template='plotly_white')
    return fig

@st.cache
def word_density(df, text_column):
    try:
        df['char_count'] = count_characters(df, text_column)
        df['word_count'] = count_words(df, text_column)
        word_density = df['char_count'] / df['word_count']+1
        return word_density
    except Exception as e:
        st.error('Must count characters and words.')
        return pd.Series()

@st.cache
def plot_word_density(word_dens, label_column):
    fig = px.box(word_dens, y='word_density', title='Word Density per Record',
                 color = label_column, template='plotly_white')
    return fig

def tag_pos(text_list):
    df['noun_count'] = text_list.apply(lambda x: check_pos_tag(x, 'noun'))
    df['verb_count'] = text_list.apply(lambda x: check_pos_tag(x, 'verb'))
    df['adj_count'] = text_list.apply(lambda x: check_pos_tag(x, 'adj'))
    df['adv_count'] = text_list.apply(lambda x: check_pos_tag(x, 'adv'))
    df['pron_count'] = text_list.apply(lambda x: check_pos_tag(x, 'pron'))
    return df

@st.cache
def word_frequency(df, text_column):
    word_dist = FreqDist(sum(df[text_column].apply(tokenize), []))
    word_dist_df = pd.DataFrame(word_dist.most_common(50), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=True)
    print(word_dist_df)
    return word_dist_df

@st.cache
def plot_word_freq(word_dist):
    fig = px.bar(word_dist, x='Frequency', y='Word', orientation='h',
                 title='Word Frequency', width=800, height=1000, template='plotly_white')
    return fig

def pipelinize(function, active=True):
    '''transforms functions to input as an estimator for SKLearn's Pipeline'''
    def list_comprehend_a_function(list_or_series, active=True):
        if active:
            return [function(i) for i in list_or_series]
        else: # if it's not active, just pass it right back
            return list_or_series
    return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})

from nltk import stem
from nltk.corpus import stopwords
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

def alternative_review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    # removing stopwords
    msg = [word for word in msg.split() if word not in stopwords]
    # using a stemmer
    msg = " ".join([stemmer.stem(word) for word in msg])
    return msg
