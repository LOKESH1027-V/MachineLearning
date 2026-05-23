#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:59:34 2026

@author: loki
"""

import numpy as np
import streamlit as st
import pickle
loaded_model=pickle.load(open('/home/loki/Developer/Ml/Project/Winequality/trained_model.sav','rb'))

def wine_quality_prediction(input_data):
    np_array=np.asarray(input_data)
    input_data_reshaped=np_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    if prediction[0]==1:
        return 'Good Quality Wine'
    else:
        return 'Low Quality Wine'



def main():
    st.title("Wine Quality Prediction")
    
    

    fixed_acidity=st.number_input("Fixed acidity")
    volatile_acidity=st.number_input("Volatile acidity")
    citric_acid=st.number_input("Citric acid")
    residual_sugar=st.number_input("Residual_sugar")
    chlorides=st.number_input("Chlorides")
    free_sulfur_dioxide=st.number_input("Free sulfur dioxide")
    total_sulfur_dioxide=st.number_input("Total sulfur dioxide")
    density=st.number_input("Density")
    pH=st.number_input("PH value")
    sulphates=st.number_input("Sulphate")
    alcohol=st.number_input("Alcohol") 
    input_data = (
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    )
    
    
    prediction=""
    
    if st.button("Predicted outcome"):
        prediction=wine_quality_prediction(input_data)
    st.success(prediction)

if __name__=='__main__':
    main()    
    