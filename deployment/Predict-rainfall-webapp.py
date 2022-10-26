# Import the required libraries for the machine learning application.
import joblib
import numpy as np
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

# Unpickle Regressor and scaler.
scaler = joblib.load('./model/scaler.pkl')
model = joblib.load('./model/rainfall_forecast.pkl')

def predict_precipitation(model, x):
    return model.predict(x)

def main():
    #st.title('Predicting the Precipitable water available for Precipitation')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predicting the Precipitable water available for Precipitation </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    omega_x = st.number_input('Enter the Horizontal Wind velocity (omega_x)')
    omega = st.number_input('Enter the Vertical Wind velocity (omega)')
    rhum_x = st.number_input('Enter the relative humidity (rhum_x)')
    rhum_y = st.number_input('Enter the relative humidity (rhum_y)')
    rhum = st.number_input('Enter the relative humidity (rhum)')
    slp = st.number_input('Enter the Mean sea level pressure (slp)')
    tmp = st.number_input('Enter Temperature')
    uwnd_x = st.number_input('Enter the x component of the wind (uwnd_x)')
    uwnd = st.number_input('Enter U-wind')

    # Store inputs into dataframe.
    X = [omega_x, omega, rhum_x, rhum_y, rhum, slp, tmp, uwnd_x, uwnd]
    X = np.array(X).reshape(1,-1)
    X = scaler.transform(X) # Transforming the input values

    # If button is pressed.
    if st.button("Predict"):
        result = predict_precipitation(model, X)
        st.success("Precitable water available for precipitation is: {:.2f} mm".format(result[0]))        


if __name__=='__main__':
    main()