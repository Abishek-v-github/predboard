import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.title("PredictBoard")

upfile = st.file_uploader("Upload CSV", type=["csv"])

if upfile is not None:
    df = pd.read_csv(upfile)
    st.write("Preview:", df.head())

    if df.isnull().values.any():
        st.warning("Warning: The dataset contains missing values. Handle them before proceeding.")

    st.sidebar.header("Handle Null Values")

    drop_columns = st.sidebar.multiselect("Drop Columns with Nulls", ["All Columns"] + df.columns.tolist())
    if "All Columns" in drop_columns:
        df = df.dropna(axis=1)
    elif drop_columns:
        df = df.drop(columns=drop_columns)

    fill_columns = st.sidebar.multiselect("Fill Null Columns", ["All Columns"] + df.columns.tolist())
    fill_strategy = st.sidebar.selectbox("Fill Strategy", ["None", "Mean", "Median", "Mode"])
    
    if fill_columns and fill_strategy != "None":
        if "All Columns" in fill_columns:
            fill_columns = df.columns.tolist()
        if fill_strategy == "Mean":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mean())
        elif fill_strategy == "Median":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].median())
        elif fill_strategy == "Mode":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mode().iloc[0])

    x_col = st.selectbox("Select X (Independent Variable)", df.columns.tolist())
    y_col = st.selectbox("Select Y (Dependent Variable)", df.columns.tolist())

    x_encode_option = st.selectbox("Encode X?", ["No Encoding", "Label Encoding", "One-Hot Encoding"])
    y_encode_option = st.selectbox("Encode Y?", ["No Encoding", "Label Encoding"])

    if x_encode_option == "Label Encoding":
        df[x_col] = LabelEncoder().fit_transform(df[x_col])

    if y_encode_option == "Label Encoding":
        df[y_col] = LabelEncoder().fit_transform(df[y_col])

    st.write("Data after encoding (if any):", df.head())

    model = None
    y_pred = None

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "KNN Regression"])
    new_val = st.number_input(f"Input a value for {x_col} to predict {y_col}", value=0.0)

    X = df[[x_col]]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if st.button("Perform Regression"):
        model = LinearRegression() if model_choice == "Linear Regression" else KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predicted_value = model.predict([[new_val]])[0]
        st.success(f"{model_choice} completed! Predicted {y_col} for {x_col} = {predicted_value:.2f}")


        accuracy = np.mean(np.round(y_pred) == y_test)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        plt.figure(figsize=(10, 6))

        if model_choice == "Linear Regression":
            plt.scatter(X_test, y_test, color="blue", label="Actual Data")
            plt.plot(X_test, y_pred, color="red", label="Regression Line")
            plt.title(f"Linear Regression: {y_col} vs {x_col}")


        elif model_choice == "KNN Regression":
            plt.scatter(X_test, y_test, color="blue", label="Actual Data")
            plt.scatter(X_test, y_pred, color="green", label="Predicted Data", alpha=0.5)
            plt.title(f"KNN Regression: {y_col} vs {x_col}")

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        st.pyplot(plt)
