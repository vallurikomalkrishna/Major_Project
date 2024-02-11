import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Function to create and display the heatmap
def display_heatmap(df):
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.title('Correlation Heatmap of the CNC Flank Wear Dataset', fontsize=20)
    sns.heatmap(df.corr(), cbar=True, cmap='plasma', annot=True, linewidths=1, ax=ax)
    st.pyplot(fig)

# Function for multiple linear regression and metrics calculation
def multiple_linear_regression_and_metrics(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, predictions, s=12, color='turquoise', label='Predicted vs Actual')
    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('Actual Values')
    ax.legend()
    st.pyplot(fig)

    # Metrics
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    r2 = metrics.r2_score(y_test, predictions)
    return model.summary(), rmse, r2

# Streamlit UI
st.title('CNC Flank Wear Prediction App')
st.markdown('Upload a CSV file to analyze the CNC Flank wear data and predict wear based on the machining parameters.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    try:
        # Data preprocessing
        data = data.rename(columns={'Machining Parameters': 'Feed rate', "Unnamed: 2": 'Depth of cut', 'Unnamed: 3': 'Speed'})
        data.dropna(inplace=True)
        data.drop(data.tail(1).index, inplace=True)  # Drop the last row
        data['Feed rate'] = pd.to_numeric(data['Feed rate'], errors='coerce')
        data['Depth of cut'] = pd.to_numeric(data['Depth of cut'], errors='coerce')
        data['Speed'] = pd.to_numeric(data['Speed'], errors='coerce')

        # Display Data and Heatmap
        if st.button('Show Data Head'):
            st.dataframe(data.head())
        if st.button('Show Heatmap'):
            display_heatmap(data)

        # Prediction Inputs
        st.sidebar.header('Set Parameters for Prediction')
        feed_rate = st.sidebar.slider('Feed Rate', float(data['Feed rate'].min()), float(data['Feed rate'].max()), float(data['Feed rate'].mean()))
        depth_of_cut = st.sidebar.slider('Depth of Cut', float(data['Depth of cut'].min()), float(data['Depth of cut'].max()), float(data['Depth of cut'].mean()))

        # Predict Button
        if st.sidebar.button('Predict Flank Wear'):
            summary, rmse, r2 = multiple_linear_regression_and_metrics(data[['Feed rate', 'Depth of cut']], data['Speed'])
            st.text('Model Summary:')
            st.text(summary)
            st.write('Metrics:')
            st.write(f'RMSE: {rmse:.2f}')
            st.write(f'R²: {r2:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info('Awaiting CSV file upload.')
