import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# companys status Prediction App
This app predicts the **Crunchbase companys** status!
Data obtained from the crunchbase realtime dataset of companys
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""Enter the values you want predict)
""")

# Collects user input features into dataframe

def user_input_features():
    founded_at = st.sidebar.slider('founded_at', 1960,2020,1960)
    first_funding_at = st.sidebar.slider('first_funding_at', 1960,2020,1960)
    last_funding_at = st.sidebar.slider('last_funding_at',1960,2020,1960)

    funding_total_usd = st.sidebar.slider('funding_total_usd', 10000.0,300000.0
,5000.0)
    first_milestone_at = st.sidebar.slider('first_milestone_at', 1960,2020,1960)
    no_of_relationships = st.sidebar.slider('relationships', 1.0,2000.0,10.0)
    funding_rounds = st.sidebar.slider('funding_rounds', 0.0,5.0,1.0)
    active_days =st.sidebar.slider('active_days', 1000.0,5000.0,10.0)
    data = {'founded_at': founded_at,
            'first_funding_at': first_funding_at,
            'last_funding_at': last_funding_at,
            'funding_rounds': funding_rounds,
            'funding_total_usd': funding_total_usd,
            'first_milestone_at': first_milestone_at,
            'relationships':no_of_relationships,
            'active_days':active_days
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


# Displays the user input features
st.subheader('User Input features')


import warnings
import joblib
with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      estimator = joblib.load(open('statusModel.sav', 'rb'))

# Apply model to make predictions
prediction = estimator.predict(df)
prediction_proba = estimator.predict_proba(df)


st.subheader('Prediction')
cleaned_company = np.array(['operating','acquired','closed','ipo'])
print(prediction)
if prediction == 4:
    prediction=3
st.write(cleaned_company[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
