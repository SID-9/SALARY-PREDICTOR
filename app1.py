import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib as plt
import altair as alt
#from  plotly import graph_objs as go

lr = LinearRegression()

st.title("Salary - Predictor")

ds = pd.read_csv("salary_data.csv")
x = np.array(ds['YearsExperience']).reshape(-1,1)
lr.fit(x,np.array(ds['Salary']))

nav = st.sidebar.radio("**Navigation**",["Home","Prediction","Contribute"])
if nav == "Home":
    st.image('media//sal.jpg')
    if st.checkbox('show data'):
        st.table(ds)
    df=pd.DataFrame(ds)
    chart=alt.Chart(df).mark_circle().encode(
    x="YearsExperience",y="Salary",).interactive()
    st.altair_chart(chart)

if nav == "Prediction":
    st.header("predict your salary")
    val = st.number_input("enter your experience",0.00,20.00,step=0.25)
    
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    if st.button("predict"):
        st.success(f"your predicted salary is :  {round(pred)}")

if nav == "Contribute":
    st.header("contribute to your dataset")
    ex = st.number_input("enter your experience : ",0.0,20.0)
    sal = st.number_input("enter your salary : ",0.00,1000000.00,step=1000.0)
    
    if st.button("submit"):
        to_add = {'YearsExperience':[ex],'Salary':[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("media//salary_data.csv",mode='a',header=False,index=False)
        st.success("submitted")


