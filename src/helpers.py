# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha
# Disclaimer: Code subject to change any time. Amruta Inc. is not responsible for any loss/damage caused by use of this code
# Purpose: Be able to load any predictive supervised learning model
# and dataset to understand what features affect your output the most
# using Explainable AI methods
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.metrics import  confusion_matrix

import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier, DMatrix
import xgboost as xgb
# from sklearn.tree import export_graphviz
# from graphviz import Source
import shap
# from dtreeviz.trees import *
import base64
import operator
from sklearn.impute import SimpleImputer
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string as s

@st.cache(hash_funcs={dict:id}, allow_output_mutation=True)
def load_data(data, separator):
    df = pd.read_csv(data, sep=separator)
    return df

@st.cache
def view_labels(df, label_column):
    value_counts = pd.DataFrame(df[label_column].value_counts()).reset_index()
    fig = px.bar(value_counts, x='index', y=label_column,
                 labels={'index':label_column, label_column:'Count'},
                 title='Value Count of Labels')
    return fig

@st.cache
def df_summary(df):
    return df.describe()

def label_encode(df, columns):
    encoded_df = df.copy()
    le = LabelEncoder()
    for col in columns:
        encoded_df[col] = encoded_df[col].astype('str')
        encoded_df[col] = le.fit_transform(encoded_df[col])
        #print(col)
        st.write(pd.DataFrame(zip(le.classes_, le.transform(le.classes_)), columns=['value', 'encoding']))
        encoding_format_df = pd.DataFrame(zip(le.classes_, le.transform(le.classes_)), columns=['value', 'encoding']).reset_index(drop = False)
    return encoded_df, encoding_format_df

def standard_scale(df, columns):
    scaled_df = df.copy()
    scaler = StandardScaler()
    scaler.fit(df[columns])
    scaled_df[columns] = scaler.transform(df[columns])
    return scaled_df

@st.cache(hash_funcs={pd.core.frame.DataFrame: id})
def process_data(cols_to_drop, cols_to_encode, df, encode_method):
    if cols_to_drop != None:
        df = df.drop(cols_to_drop, axis=1)

    if cols_to_encode != None:
        if encode_method == 'Label Encoder':
            df = label_encode(df, cols_to_encode)
        elif encode_method == 'One Hot Encoder':
            df = pd.get_dummies(df, columns = cols_to_encode)

    return df


@st.cache
def calc_corr(df):
    return df.corr()

def render_heatmap(df):
    """displays correlation matrix in heatmap format using Seaborn"""

    cor = calc_corr(df)
    if cor.shape[0] < 15:
        plt.figure(figsize=(30,30))
        sns.heatmap(cor, cmap='YlGnBu', annot=True, annot_kws={"size": 12})
        st.pyplot(bbox_inches='tight')
    else:
        st.write(cor)



#########################################################
################### EXPLAINABILITY ######################
#########################################################


###### LIME ######
@st.cache(suppress_st_warning=True)
def lime_explainability(model, xtest, ytest, yscore, pred_var):
    categorical_features = np.argwhere(np.array([len(set(xtest.values[:,x]))
                                                 for x in range(xtest.values.shape[1])]) <= 20).flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(xtest.values,
                                                       feature_names = xtest.columns.values.tolist(),
                                                       class_names = [pred_var],
                                                       categorical_features=categorical_features,
                                                       verbose = True,
                                                       mode = 'classification')

    return explainer



def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="3000" height="3000"/>' % b64
    st.write(html, unsafe_allow_html=True)


def tokenization(text):
    lst=text.split()
    return lst


## Conversion of Data to Lowercase

def lowercasing(lst):
    new_lst=[]
    for  i in  lst:
        i=i.lower()
        new_lst.append(i) 
    return new_lst

## Removal of Punctuation Symbols

def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for  j in  s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst

## Removal of Numbers(digits)

def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]

    for i in  lst:
        for j in  s.digits:
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in  nodig_lst:
        if  i!='':
            new_lst.append(i)
    return new_lst

## Removal of Stopwords


def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst


## Lemmatization of Data

lemmatizer=WordNetLemmatizer()
def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst


def data_preprocessor_function(df,inp_column):
    inp_df = df.copy()
    inp_df[inp_column] = inp_df[inp_column].apply(tokenization)
    inp_df[inp_column] = inp_df[inp_column].apply(lowercasing)
    inp_df[inp_column] = inp_df[inp_column].apply(remove_punctuations)
    inp_df[inp_column] = inp_df[inp_column].apply(remove_numbers)
    inp_df[inp_column] = inp_df[inp_column].apply(remove_stopwords)
    inp_df[inp_column] = inp_df[inp_column].apply(lemmatzation)
    inp_df[inp_column] = inp_df[inp_column].apply(lambda x: ''.join(i+' '  for i in x))
    return inp_df


def eval_model(y,y_pred):
    st.write("F1 score of the model")
    st.write(f1_score(y,y_pred,average='micro'))
    st.write("Accuracy of the model")
    st.write(accuracy_score(y,y_pred))
    st.write("Accuracy of the model in percentage")
    st.write(round(accuracy_score(y,y_pred)*100,3),"%")

def confusion_mat(color,test_y,pred):
    cof=confusion_matrix(test_y, pred)

    sns.heatmap(cof, cmap=color, linewidths=1, annot=True, square=True, fmt='d', cbar=False);
    plt.xlabel("Predicted Classes")
    plt.ylabel("Actual Classes")
    st.pyplot()


def train(clf, features, targets):
    clf.fit(features, targets)

def predict(clf, features):
    return (clf.predict(features))

