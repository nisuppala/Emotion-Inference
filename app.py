"""
AMRUTA OXAI V2
Copyright Amruta Inc. 2021


Features:
* Upload your own dataset
* Dataset info and summary
* Train/Test Split
* Modeling
    - Logistic Regression
    - Random Forest
    - Decision Tree
    - XGBoost
    - Light GBM
    - CatBoost
* Model Evaluation
    - Regression
        -
    - Classification
        - Accuracy, Precision, Recall, F1
        - Confusion Matrix
        - ROC/AUC Curve
* Explainability
    - SHAP
    - feature importance
    - tree visualizer
    - correlation graph
    - tree interpreter
    - ice plots
    - pdp plots

Features to add:
* fill in missing data
* encode categorical variables - DONE
* Work with Multiclass classfication - DONE
* Correlation graph - DONE
* ICE Plots = DONE
* SFIT (Single Feature Introduction Test)
* connect to dataset hosted on cloud: GCP, Azure, AWS
* user login/authentication - DONE
* add info based on user selected input to sidebar
* add light gbm

TODO:
* need to define hash function for XGBRegressor/XGBClassifier under SessionState - DONE
* Categorize Explainability methods - DONE
    - Global Summary
    - Local Explanation
    - PDP/Interaction Plots
    - Feature Importance
"""
import regex as re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from src.nlp_functions import *
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from src.helpers import *
from src.modeling import *
from src.shap_exp import *
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.metrics import precision_recall_fscore_support as score
import scipy as sp
#from src.ice_pdp import *
#from src.correlation_graph import *
#from src.lime_eli5 import *
#from src.feat_importance import *
#from sfit import *
from datetime import datetime, date
import streamlit.components.v1 as components
import shap
from src.SessionState import *
from src.account_management import validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import eli5
import streamlit.components.v1 as components
import lime
from lime.lime_text import LimeTextExplainer
import sklearn.ensemble
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfTransformer

# set seaborn theme
sns.set()

# # st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)


SAMPLE_NEWS_DATA = 'sample_data/news_classification_updated.csv'
LOGO_IMAGE = 'images/amrutalogo2.png'
UPLOADED_DATASET = None
UPLOADED_MODEL = None

# declare session state
state = get_state({XGBClassifier:id, XGBRegressor:id})
#state.model = None
#state.vectorizer = None
#state.X_train = None
#state.X_test = None
#state.text_column = None
#state.label_column = None

def render_app():
    st.sidebar.title('Amruta XAI V3.5.1')

    ####### PAGE SELECTION ########
    pages = {
        'Data Explorer': data_exploration,
        'Data Processor': feature_engineering,
        'ML Modeler': modeling,
        'ML Explainer': explainability,
        'Emotion Inference': emotion_inference,
        'Notes': notes
    }

    sample_data_select = st.sidebar.selectbox('Select Sample data:', ['News Article Topics','None'])

    dataset_shape = st.sidebar.empty()
    separator = st.sidebar.selectbox('Select separator used in your dataset', ['\t', ',', '|', '$', ' '], 1)
    UPLOADED_DATASET = st.sidebar.file_uploader('Upload your preprocessed dataset in CSV format (Size Limit: 1028mb)', type='csv')

    #UPLOADED_MODEL = st.sidebar.file_uploader('And/Or upload your pre-trained model (.sav or .pkl) (Size Limit: 1028mb)', type=['sav', 'pkl'])
    UPLOADED_MODEL = None

    tab = st.sidebar.radio('Select Tab', list(pages.keys()))
    #st.sidebar.image(Image.open(LOGO_IMAGE), width = 300)
    #st.sidebar.text('Copyright Amruta Inc. 2021')
    #st.sidebar.text('Beta/Test Version')
    #st.sidebar.text('The purpose of this version is to test \nnew features.')
    #st.sidebar.text("Logged in as %s" % state.user_name)

    #log_out = st.sidebar.button('Log out')
    #if log_out:
    #    state.user_name = None

    ## dataset selection
    df = None
    if UPLOADED_DATASET is not None:
        UPLOADED_DATASET.seek(0)
        sample_data_select = 'None'
        data_load_state = st.text('Loading data...')
        data_to_load = UPLOADED_DATASET
        data_load_state.text('Loading data... done!')

    else:
        if sample_data_select == 'News Article Topics':
            data_to_load = SAMPLE_NEWS_DATA
        else:
            st.info('Please select a sample dataset or upload a dataset.')
            st.stop()

    ### LOAD DATA
    if ((df is None) and (state.processed_df is None)) or ((state.current_fn != data_to_load) and (UPLOADED_DATASET is None)):
        if data_to_load:
            try:
                df = load_data(data_to_load, separator)
                state.processed_df = None
                state.current_fn = data_to_load

            except FileNotFoundError:
                st.error('File not found.')
            except:
                st.error('Make sure you uploaded the correct file format.')
        else:
            st.info('Please upload some data or choose sample data.')

    # try:
    #     state.text_column = st.sidebar.selectbox('Which column contains text?', df.columns)
    #     state.label_column = st.sidebar.selectbox('Which column contains label?', df.columns)
    # except:
    #     state.text_column = st.sidebar.selectbox('Which column contains text?', ['None'])
    #     state.label_column = st.sidebar.selectbox('Which column contains label?', ['None'])
    ### LOAD MODEL
    user_model = None
    if UPLOADED_MODEL is not None:
        try:
            UPLOADED_MODEL.seek(0)
            user_model = load_model(UPLOADED_MODEL)
        except Exception as e:
            st.error(e)

    ## view dataset rows and columns in sidebar
    if state.processed_df is None:
        dataset_shape.text('Dataset shape\n Rows: %s\n Columns:%s' % (str(df.shape[0]), str(df.shape[1])))
    else:
        dataset_shape.text('Dataset shape\n Rows: %s\n Columns:%s' % (str(state.processed_df.shape[0]), str(state.processed_df.shape[1])))
    ########## TAB SELECTION ############
    if tab == 'Data Explorer':
        if (state.processed_df is not None) and not (state.processed_df.equals(df)):
            data_exploration(state.processed_df)
        else:
            data_exploration(df)
    elif tab == 'Data Processor':
        if state.processed_df is not None:
            state.processed_df,state.label_map = feature_engineering(state.processed_df)
        else:
            state.processed_df,state.label_map = feature_engineering(df)
    elif tab == 'ML Modeler':
        if state.processed_df is not None:
            try:
                #state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred = modeling(state.processed_df, user_model)
                state.model, state.vectorizer, X_train, X_test, state.export_data = modeling(state.processed_df)
            except TypeError as e:
                print(e)
        else:
            try:
                #state.model, state.xtrain, state.xtest, state.ytrain, state.ytest, state.ypred = modeling(df, user_model)
                state.model, state.vectorizer, X_train, X_test, state.export_data = modeling(df)
            except TypeError as e:
                pass
    elif tab == 'ML Explainer':
        if state.processed_df is not None:
            explainability(state.model, state.vectorizer, state.X_train, state.X_test, state.export_data,state.label_map)
            pass
        else:
            explainability(state.model, state.vectorizer, state.X_train, state.X_test, state.export_data,state.label_map)
            pass
    elif tab == 'Emotion Inference':
        if state.processed_df is not None:
            emotion_inference(state.processed_df)
            pass
        else:
            emotion_inference(df)
            pass
    elif tab == 'Notes':
        notes()



def data_exploration(df):


    st.header('Data Explorer')
    text_column = st.selectbox('Which column contains text?', df.columns)
    label_column = st.selectbox('Which column contains label?', df.columns)


    explore_options = ['View whole dataset', 'Single Record', 'Word Frequency', 'Label Count',
                       'Character Count Distribution', 'Word Count Distribution', 'Word Density Distribution']
    state.explore_select = st.selectbox('View:', explore_options,
                                         explore_options.index(state.explore_select) if state.explore_select else 0)
    ## view data
    if state.explore_select == 'View whole dataset':
        rows, cols = df.shape
        st.markdown('**%d** records'%rows)
        st.table(df)

    ## look at each record
    elif state.explore_select == 'Single Record':
        st.subheader('A Look at a Record')
        state.record_i = st.slider('Select Record index', min_value=0, max_value=df.shape[0]-1, value = state.record_i)
        st.text(df.iloc[state.record_i].loc[text_column])
        st.markdown('Label: **%s**'%df.iloc[state.record_i].loc[label_column])
        sr_button = st.checkbox('*Description*')
        if sr_button:
            st.write('Shows just the chosen row of the dataset.')

    ## word frequency
    elif state.explore_select == 'Word Frequency':
        word_dist = word_frequency(df, text_column)
        wd_fig = plot_word_freq(word_dist)
        st.plotly_chart(wd_fig, bbox_inches='tight')
        wf_button = st.checkbox('*Description*')
        if wf_button:
            st.write('The 50 words that appear the most frequently in the chosen text column.')

    ## labels
    elif state.explore_select == 'Label Count':
        vc_fig = view_labels(df, label_column)
        st.plotly_chart(vc_fig, bbox_inches='tight')
        lc_button = st.checkbox('*Description*')
        if lc_button:
            st.write('The frequency of the labels in the dataset.')

    ## box plot character count and word count
    elif state.explore_select == 'Character Count Distribution':
        count_df = df.copy()
        char_count = count_characters(count_df, text_column)
        count_df['char_count'] = char_count
        char_fig = plot_char_count(count_df, label_column)
        st.plotly_chart(char_fig, bbox_inches='tight')
        ccd_button = st.checkbox('*Description*')
        if ccd_button:
            st.write('The average amount of characters per message of each category.')

    elif state.explore_select == 'Word Count Distribution':
        count_df = df.copy()
        word_count = count_words(count_df, text_column)
        count_df['word_count'] = word_count
        word_fig = plot_word_count(count_df, label_column)
        st.plotly_chart(word_fig, bbox_inches='tight')
        wcd_button = st.checkbox('*Description*')
        if wcd_button:
            st.write('The average amount of words per message of each category.')

    elif state.explore_select == 'Word Density Distribution':
        count_df = df.copy()
        word_dens = word_density(count_df, text_column)
        count_df['word_density'] = word_dens
        word_dens_fig = plot_word_density(count_df, label_column)
        st.plotly_chart(word_dens_fig, bbox_inches='tight')
        wdd_button = st.checkbox('*Description*')
        if wdd_button:
            st.write('Character count divided by word count.')



def feature_engineering(df):
    st.markdown('''
    Click on a drop down option to implement pre-processing steps.
    ''')
    processed_df = df.copy()

    ## index/slicing
    #st.subheader('Index/Slice DataFrame')

    with st.beta_expander('Index/Slice DataFrame'):
        view_desc1_button = st.button('View Description', key=301)
        if view_desc1_button:
            '''
            Subset dataframe by its index. For example, for a dataset of size 100, choosing
            10 as the head index and 100 as the tail index will ignore the first 10 records. Choosing
            0 as the head index and 50 as the tail index will only subset the first 50 records.
            '''

        print(state.slice_values)
        state.slice_values = st.slider('Select index range',
                                    min_value=0,
                                    max_value=df.shape[0],
                                    value=state.slice_values if state.slice_values
                                    and (0 <= state.slice_values[0] <= df.shape[0])
                                    and (state.slice_values[0] <= state.slice_values[1] <= df.shape[0])
                                    else [0, df.shape[0]])

        slice_button = st.checkbox('Slice')
        if slice_button:
            try:
                df_sliced = df.iloc[state.slice_values[0]:state.slice_values[1]]
                st.success('DataFrame sliced!')
            except:
                st.error('Dataframe unable to slice.')
        else:
            df_sliced = df
    #--------------------------------------------------------------------------------------------------#

    ##drop columns
    # st.subheader('Drop any columns')
    with st.beta_expander('Drop Columns'):
        view_desc2_button = st.button('View Description', key=302)
        if view_desc2_button:
            '''
            Drop any columns/features that are unnecessary for fitting the model. This may include ID, names, and any
            other features that serve as identification or are redundant.
            '''

        state.drop_cols = st.multiselect('Select columns to drop.',
                                        list(df_sliced.columns),
                                        state.drop_cols
                                        if state.drop_cols in list(df_sliced.columns)
                                        else None)

        drop_button = st.checkbox('Drop')
        if drop_button:
            try:
                df_dropped = df_sliced.drop(state.drop_cols, axis=1)
                if len(state.drop_cols) > 0:
                    st.success('Column(s) %s dropped!'%(str(state.drop_cols)))
            except:
                st.error('Failed to drop columns in dataframe.')
        else:
            df_dropped = df_sliced
    #---------------------------------------------------------------------------------------------------#
    ## Null Value Imputation

    # choose which columns to impute

    # for each column/feature, choose imputation method
    # must choose whether it must impute continuous values or categorical

    with st.beta_expander('Apply Preprocessing'):
        st.write('''
            Preprocessing will clean the Input text data.

            1. Remove the white spaces.

            2. Convert text to lower case.

            3. Remove the stopwords (Stop words are common words which do not add predictive value because they are found everywhere.)

            4. Stem the words (Stemming/lemmatizing words attempts to get the root word for different word inflections. For example: Raining -> Rain')
                ''')
        state.select_process_column = st.selectbox('Select text column to Apply Preprocessing.',list(df_dropped.columns))
        state.Preprocessing_button = st.checkbox('Preprocessing ', state.Preprocessing_button)
        if state.Preprocessing_button:
            #try:
            df_dropped = data_preprocessor_function(df,state.select_process_column)
            st.success('Preprocessing Applied!')
            #except:
                #st.error('Make sure you chose the right column in Preprocessing.')
        else:
            df_dropped = df_dropped

    #---------------------------------------------------------------------------------------------------#

    # feature generation
    # with st.beta_expander('Feature Generation / Prepare data for Model'):
    #     feature_df = df_dropped.copy()
    #     view_desc3_button = st.button('View Description', key=303)
    #     if view_desc3_button:
    #         '''
    #         Bag-of-words model(BoW ) is the simplest way of extracting features from the text. BoW converts text into the matrix of occurrence of words within a document.
    #         This model concerns about whether given words occurred or not in the document.

    #         TF-IDF vectorizes documents by calculating a TF-IDF statistic between the document and each term in the vocabulary.
    #         The document vector is constructed by using each statistic as an element in the vector.

    #         Please select the processed data for optimum feature generation.
    #         '''
    #     #feature_options = ['tf-idf','bag-of-words']
    #     #state.feature_select = st.selectbox('View:', feature_options)
    #     state.generate_feature_column = st.selectbox('Select text column to Generate Features.',
    #                                     list(df_dropped.columns))
    #     state.feature_button = st.checkbox('Generate Features', state.feature_button)
    #     if state.feature_button:
    #         feature_df['Processed_text'] = text_feature_tfidf(df_dropped,state.generate_feature_column)


        # if state.feature_button:
        #     if feature_options == 'tf-idf':
        #         # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        #         # cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
        #         # feature_df['Processed_text']= cv.fit_transform(df_dropped[state.generate_feature_column])
        #         try:
        #             feature_df['Processed_text'] = text_feature_tfidf(df_dropped[state.generate_feature_column])
        #             st.success('Features Generated!')
        #         except error as e:
        #             st.write(e)
        #     elif feature_options == 'bag-off-words':
        #         #tf=TfidfVectorizer()
        #         #feature_df['Processed_text']= tf.fit_transform(df_dropped[state.generate_feature_column])
        #         st.success('Features Generated!')
        #     else:
        #         st.error('unable to create Features.')
        #         feature_df = df_dropped


    # #---------------------------------------------------------------------------------------------------#

    ## encode features
    # st.subheader('Label Encode columns')

    with st.beta_expander('Label Encoding'):
        feature_df = df_dropped
        view_desc4_button = st.button('View Description', key=304)
        if view_desc4_button:
            '''
            Certain categorical columns that contain text attributes will need to be encoded into integer values. XAI V2
            provides 2 methods of encoding: Label Encoding and One-Hot Encoding.


            **Label Encoding** will transform each text value into an integer value and output a table of what
            the integer encoding stands for.

            **One-Hot Encoding** will transform each text value into its own separate column and label
            as 0 or 1. If the record contains that text value, it will be marked as 1, otherwise 0. This is recommended
            for features with high number of categories. Keep in mind that One-Hot Encoding could drastically increase
            the dimensionality of your dataset, which would affect model training time.
            '''

        encode_meth_options = ['Label Encoder', 'One Hot Encoder']
        state.encode_method_select = st.selectbox('Select Encoding Method',
                                            encode_meth_options,
                                            encode_meth_options.index(state.encode_method_select) if state.encode_method_select else 0)
        state.cols_to_encode = st.multiselect('Select Columns to encode',
                                            list(feature_df.columns), state.cols_to_encode if state.cols_to_encode in list(feature_df.columns) else None)
        encode_button = st.checkbox('Encode')
        if encode_button:
            try:
                if state.encode_method_select == 'Label Encoder':
                    df_encoded,state.encoding_format_df = label_encode(feature_df, state.cols_to_encode)
                elif state.encode_method_select == 'One Hot Encoder':
                    df_encoded,state.encoding_format_df = pd.get_dummies(feature_df, columns = state.cols_to_encode)
                st.success('Column(s) %s encoded!'%(str(state.cols_to_encode)))
            except:
                st.error('Failed to encode columns.')
        else:
            df_encoded = feature_df


    #---------------------------------------------------------------------------------------------------#

    ## standard scaling numeric values
        # st.subheader('Standard Scaler')

    # with st.beta_expander('Continuous Feature Normalization'):
    #     view_desc5_button = st.button('View Description', key=305)
    #     if view_desc5_button:
    #         '''
    #         Standardize features by removing the mean and scaling to unit variance

    #         The standard score of a sample x is calculated as:
    #         z = (x - u) / s
    #         Standardization of a dataset is a common requirement for many machine learning estimators:
    #         they might behave badly if the individual features do not more or less look like standard
    #         normally distributed data (e.g. Gaussian with 0 mean and unit variance).
    #         '''

    #     state.cols_to_scale = st.multiselect('Select Columns to scale',
    #                                         list(df_encoded.columns),
    #                                         state.cols_to_scale if state.cols_to_scale in list(df_encoded.columns)
    #                                         else None)

    #     scale_button = st.checkbox('Scale')
    #     if scale_button:
    #         try:
    #             df_scaled = standard_scale(df_encoded, state.cols_to_scale)
    #             st.success('Column(s) %s Scaled!'%(str(state.cols_to_scale)))
    #         except:
    #             st.error('Failed to standard scale columns.')
    #     else:
    #         df_scaled = df_encoded

    st.subheader('Processed Data Preview')
    st.write(df_encoded.head(20))
    download_processed_data(df_encoded)

    process_confirm_button = st.button('Confirm Processing Steps')
    if process_confirm_button:
        processed_df = df_encoded
        st.success('Processing steps applied!')

    return processed_df,state.encoding_format_df

    #---------------------------------------------------------------------------------------------------#



def modeling(df):
    # classifier = Pipeline([
    #     ('features', FeatureUnion([
    #         ('text', Pipeline([
    #             ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, stop_words=stop_words,
    #                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
    #             ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
    #             ('svd', TruncatedSVD(algorithm='randomized', n_components=300)), #for XGB
    #         ])),
    #         ('words', Pipeline([
    #             ('wordext', NumberSelector('TotalWords')),
    #             ('wscaler', StandardScaler()),
    #         ])),
    #     ])),
    #     ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
    # #    ('clf', RandomForestClassifier()),
    #     ])
    export_data = df
    model = None
    X_train = None
    X_test = None
    vectorizer = None
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(multi_class='multinomial')
    rfc = RandomForestClassifier(n_estimators=10)
    lgbm = LGBMClassifier(num_leaves=31, n_estimators=10)
    clfs = {'Support Vector Classification': svc, 'K Neighbors Classifier': knc, 'Naive Bayes': mnb,
            'Decision Trees': dtc, 'Logistic Regression': lrc, 'Random Forest': rfc, 'LGBM': lgbm}

    st.header('Modeling')
    fe = st.selectbox('Choose Feature Engineering', ('Count Vectorizer', 'TFIDF'))
    algo = st.selectbox('Choose Model', ('Support Vector Classification','K Neighbors Classifier', 'Naive Bayes', 'Decision Trees', 'Logistic Regression', 'Random Forest', 'LGBM'))
    text_column = st.selectbox('Select the text column',df.columns,key = 101)
    label_column = st.selectbox('Select the label column',df.columns,key = 102)

    if fe == 'Count Vectorizer':
        vectorizer = CountVectorizer()
    elif fe == 'TFIDF':
        vectorizer = TfidfVectorizer()

    if st.button("GO"):
        vectors = vectorizer.fit_transform(df[text_column])
        X_train, X_test, y_train, y_test = train_test_split(vectors, df[label_column], test_size=0.15, random_state=111)
        model = clfs[algo]

        train(model, X_train, y_train)
        y_pred = model.predict(X_test)
        #precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')
        #st.write('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test),3)))

        state.model = model
        state.vectorizer = vectorizer
        state.X_test = X_test
        state.X_train = X_train

        st.write(eval_model(y_test, y_pred))
        confusion_mat('Reds', y_test, y_pred)
        #st.pyplot(bbox_inches='tight')


    return model,vectorizer, X_train, X_test, export_data

def explainability(model, vectorizer, X_test, X_train, export_data,label_map):
    st.header('Explainability')

    exp = st.selectbox('Select Explainability method to use:', ('ELI', 'SHAP', 'Lime'))

    if exp == 'ELI':
        if st.button("GO"):
            eli5_plot = eli5.show_weights(model, vec=vectorizer, top=20)
            components.html(eli5_plot.data.replace("\n", ""),width=1000, height=1000, scrolling=True)

    elif exp == 'SHAP':
        if st.button("GO"):
            X_train_sample = shap.sample(state.X_train, 100)
            X_test_sample = shap.sample(state.X_test, 20)

            SHAP_explainer = shap.KernelExplainer(model.predict, X_train_sample)
            shap_vals = SHAP_explainer.shap_values(X_test_sample)
            colour_test = pd.DataFrame(X_test_sample.todense())
            try:
                st.pyplot(shap.summary_plot(shap_vals, colour_test, feature_names=vectorizer.get_feature_names()))
            except:
                shap.summary_plot(shap_vals, colour_test, feature_names=vectorizer.get_feature_names())
                st.pyplot(bbox_inches='tight')

    elif exp == 'Lime':
        #text_column = st.selectbox('Select the text column', state.export_data.columns, key=103)
        #label_column = st.selectbox('Select the label column', state.export_data.columns, key=104)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(state.export_data['VALUE'], state.export_data['trust'], test_size=0.30, random_state=42)

        c = make_pipeline(vectorizer, model)

        ls_X_test = list(X_test)
        class_names = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6}

        LIME_explainer = LimeTextExplainer(class_names=class_names)
        state.idx = st.slider('Select Record index', min_value=0, max_value=500, value = state.idx)
        LIME_exp = LIME_explainer.explain_instance(ls_X_test[state.idx], c.predict_proba, num_features=6, labels=[0, 1,2,3,4,5,6])

        st.write('Document id: %d' % state.idx)
        st.write('Text: ', ls_X_test[state.idx])

        #st.graphviz(LIME_exp)
        #st.pyplot(bbox_inches='tight')
        try:
            html = LIME_exp.as_html()
            components.html(html, height=800)
        except:
            st.write('Can not output Lime values.')

def emotion_inference(df):
    st.header('Emotion Inference')

    df = df

    ei_type = st.selectbox('Choose type of text data', ('Individual Text', 'Dataset'))

    def emotion_predict(chosen_emotion, input_text):
        X_train, X_test, y_train, y_test = train_test_split(df['VALUE'], df[chosen_emotion], random_state=0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
        return clf.predict(count_vect.transform([input_text]))

    if ei_type == 'Individual Text':
        input_text = st.text_area("Type text here.")

        df_emotion = pd.DataFrame()
        df_emotion['anger'] = emotion_predict('anger', input_text)
        df_emotion['anticipation'] = emotion_predict('anticipation', input_text)
        df_emotion['disgust'] = emotion_predict('disgust', input_text)
        df_emotion['fear'] = emotion_predict('fear', input_text)
        df_emotion['joy'] = emotion_predict('joy', input_text)
        df_emotion['sadness'] = emotion_predict('sadness', input_text)
        df_emotion['surprise'] = emotion_predict('surprise', input_text)
        df_emotion['trust'] = emotion_predict('trust', input_text)

        if st.button("GO"):
            st.write(df_emotion)

    elif ei_type == 'Dataset':
        uploaded_file = st.file_uploader('Upload your preprocessed dataset in CSV format (Size Limit: 1028mb)', type='csv', key = 101)
        if uploaded_file is not None:
            uploaded_file.seek(0)
            df1 = pd.read_csv(uploaded_file)
            st.write(df1.head(5))

            if st.button("GO"):
                df1['anger'] = df1.apply(lambda x: emotion_predict('anger', x[0]), axis=1)
                df1['anticipation'] = df1.apply(lambda x: emotion_predict('anticipation', x[0]), axis=1)
                df1['disgust'] = df1.apply(lambda x: emotion_predict('disgust', x[0]), axis=1)
                df1['fear'] = df1.apply(lambda x: emotion_predict('fear', x[0]), axis=1)
                df1['joy'] = df1.apply(lambda x: emotion_predict('joy', x[0]), axis=1)
                df1['sadness'] = df1.apply(lambda x: emotion_predict('sadness', x[0]), axis=1)
                df1['surprise'] = df1.apply(lambda x: emotion_predict('surprise', x[0]), axis=1)
                df1['trust'] = df1.apply(lambda x: emotion_predict('trust', x[0]), axis=1)
                st.write(df1)
                download_processed_data(df1)






def notes():
    st.header('Notes')
    '''
    \n
    When working with more than one uploaded dataset, you may need to refresh the application again to switch from
    one dataset to another.
    '''


#### MAIN ####
def main():
    render_app()


main()
