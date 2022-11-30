import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import streamlit as st

st.write("""
# Red Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")

st.sidebar.header('User Input Parameters')


def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.6, 15.9, 8.31)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.52)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.5)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.0, 5.0, 1.5)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.6, 0.08)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 20.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.0)
    density = st.sidebar.slider('Density', 0.99007, 1.00369, 0.99123,0.0001)
    pH = st.sidebar.slider('pH', 2.74, 4.01, 3.0)
    sulphates = st.sidebar.slider('Sulphates', 0.33, 2.0, 0.65)
    alcohol = st.sidebar.slider('Alcohol', 8.4, 14.9, 10.0)
    data = {'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar' : residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}
    features = pd.DataFrame(data, index=[0])
    return features


df1 = user_input_features()
st.subheader('User Input parameters')
st.write(df1)


df = pd.read_csv("winequality-red.csv")

mmc = MinMaxScaler()
x = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values


train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.15)

train_x = mmc.fit_transform(train_x)
test_x = mmc.fit_transform(test_x)
svr = SVR(gamma="scale", kernel="rbf")

svr.fit(train_x, train_y)

st.subheader('Wine quality labels and their corresponding index number')
st.write(pd.DataFrame({'wine quality': [3, 4, 5, 6, 7, 8]}))


pred_y = svr.predict(df1)

st.subheader('Prediction')
st.write(pred_y)
