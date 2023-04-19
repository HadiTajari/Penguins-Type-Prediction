import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Penguin Prediction App
this is a simple app to predict penguin type 
#  """)
st.markdown("""
[Orginal Dataset Link](https://github.com/allisonhorst/palmerpenguins) """)

st.sidebar.header("Input Features")
st.sidebar.markdown("""
[Example CSV inputs file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv) """)

uploaded_file = st.sidebar.file_uploader("Upload your input csv file", type= ["csv"])

if uploaded_file is not None:
    input_user = pd.read_csv(uploaded_file)
else:   

    def input_user_features():
        sex = st.sidebar.selectbox("sex", ("male", "female"))
        island = st.sidebar.selectbox("island", ("Biscoe", "Dream", "Torgersen"))
        bill_length_mm = st.sidebar.slider("bill_length_mm", 32.0, 60.0, 41.0)
        bill_depth_mm = st.sidebar.slider("bill_depth_mm", 13.0, 22.0, 15.0)
        flipper_length_mm = st.sidebar.slider("flipper_length_mm", 170, 235, 200)
        body_mass_g = st.sidebar.slider("body_mass_g", 2700, 6300, 3500)
        data = {"island": island, 
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_depth_mm,
                "flipper_length_mm":flipper_length_mm,
                "body_mass_g": body_mass_g,
                "sex": sex,}
        features = pd.DataFrame(data, index=[0])
        return features
    input_user = input_user_features()

#importing raw dataset to encondig for new sample
penguins_raws = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raws.drop("species" ,axis=1)

df = pd.concat([input_user, penguins], axis=0)

# selecting "Object" type feature
encod_col = df.select_dtypes("O").columns.values
for col in encod_col:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]   


st.subheader("Input User Features")
if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploded. Currently using example input parametrs.(Shown below)")
    st.write(df)

# importing fitted model
model_pickle = pickle.load(open ("penguins_clf.pkl", "rb"))
prediction_df = model_pickle.predict(df)
prediction_df_prob = model_pickle.predict_proba(df)

st.subheader("Prediction type of penguin")
st.write(  prediction_df   )

st.subheader("Prediction Probability of  Penguin Types")
st.write(  prediction_df_prob   )
