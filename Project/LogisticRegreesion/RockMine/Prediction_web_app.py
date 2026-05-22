#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 06:35:13 2026

@author: loki
"""

import  numpy as np
import streamlit as st
import pickle
load_model=pickle.load(open('/home/loki/Developer/Ml/Project/LogisticRegreesion/RockMine/Trainedmodel.sav','rb'))

def rock_mine_predict(input_data):

    #converting the input into np array
    np_array=np.asarray(input_data)
    #reshaping
    np_array_reshaped=np_array.reshape(1,-1)
    ans=load_model.predict(np_array_reshaped)
    if ans=='R':
        return ("The object is Rock")
    else:
        return ("The object is Mine")    

def main():
    
    st.title("Rock Mine prediction Web App")
    
    
    input_data = []
    
    col1, col2, col3 = st.columns(3)
    
    for i in range(60):
        value = st.number_input(f"Feature {i+1}", format="%.4f")
        input_data.append(value)
    
    result=''
    if st.button("Result of the prediction"):
        result=rock_mine_predict(input_data)
    st.success(result)


if __name__== '__main__':
    main()    
        