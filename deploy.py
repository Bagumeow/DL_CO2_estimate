import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

model_cnn = tf.keras.models.load_model('data_preprocess/dillated_cnn')
look_ahead = 49

def split(df):    
    X = df[:-look_ahead].T
    Y = df[-look_ahead:].T
    X = X.values[..., np.newaxis]
    Y = Y.values[..., np.newaxis]
    return X,Y 
df = pd.read_csv("data_preprocess/df_preprocess.csv",index_col=0)

st.title('Predicting CO2 Emissions in Rwanda')
column_names = df.columns.tolist()

# Select a column
selected_column = st.selectbox("Choose a column:", column_names)
# Get values of the selected column and save to a list
print(selected_column)
data_test = df[[c for c in df.columns if c in [selected_column]]] 
x_test,y_test = split(data_test)
preds = model_cnn.predict(x_test).squeeze()

plt.plot(range(0,len(x_test[0])),x_test[0,:,0],color='green',label='Before 2021')
plt.plot(range(len(x_test[0]),len(x_test[0])+len(preds)),preds,color='orange',label='Predicted 2021')
plt.xlabel('Weeks')
plt.ylabel('Values')
plt.title(f'Predicted "{selected_column}" values over 2019 to 2021')
plt.legend()
plt.grid(True)
st.pyplot(plt)