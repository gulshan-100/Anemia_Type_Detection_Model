import streamlit as st
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

model_path = r'C:\Users\DELL\Documents\Anemia_classification\Model\anemia_classification.pkl'
model = joblib.load(model_path)

st.header('ANEMIA CLASSIFICATION AI MODEL')

WBC = st.text_input("Enter WBC")
LYMp = st.text_input('Enter LYMp')
NEUTp = st.text_input('Enter NEUTp')
LYMn = st.text_input('Enter LYM,')
NEUTn = st.text_input('Enter NEUTn')
RBC = st.text_input('Enter RCB')
HGB = st.text_input('Enter HGB')
HCT = st.text_input('Enter HCT')
MCV = st.text_input('Enter MCV')
MCH = st.text_input('MCH')
MCHC = st.text_input('Enter MCHC')
PLT = st.text_input('Enter PLT')
PDW = st.text_input('Enter PDW')
PCT = st.text_input('Enter PCT')

# Predict button
if st.button("Predict"):
    input_data = (WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT)
    input_data_array = np.asarray(input_data, dtype = float)
    input_data_reshaped = input_data_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)

    label_mapping = {
            0: 'Healthy',
            1: 'Iron deficiency anemia',
            2: 'Leukemia',
            3: 'Leukemia with thrombocytopenia',
            4: 'Macrocytic anemia',
            5: 'Normocytic hypochromic anemia',
            6: 'Normocytic normochromic anemia',
            7: 'Other microcytic anemia',
            8: 'Thrombocytopenia'
    }

    st.write(f'The prediction is: {label_mapping[prediction[0]]}')