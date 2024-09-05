import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("PredictBoard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Preview :", df.head())

    st.sidebar.header("Handle Null Values")

    drop_columns = st.sidebar.multiselect(
        "Select Columns to Drop if They Contain Nulls",
        options=df.columns.tolist()
    )

    if drop_columns:
        df = df.drop(columns=drop_columns)

    fill_columns = st.sidebar.multiselect(
        "Select Columns to Fill Nulls",
        options=df.columns.tolist()
    )

    fill_strategy = st.sidebar.selectbox(
        "Choose Fill Strategy for Selected Columns",
        ["None", "Fill with Mean", "Fill with Median", "Fill with Mode"]
    )

    if fill_columns and fill_strategy != "None":
        if fill_strategy == "Fill with Mean":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mean())
        elif fill_strategy == "Fill with Median":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].median())
        elif fill_strategy == "Fill with Mode":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mode().iloc[0])

    st.write("Updated Data Preview", df.head())

    if df.empty:
        st.error("Dataframe is empty after null value handling. Please upload another csv")
    else:
        column_options = df.columns.tolist()
        x_col = st.selectbox("Select X (Independent Variable)[numeric]", column_options)
        y_col = st.selectbox("Select Y (Dependent Variable)[numeric]", column_options)

        model_choice = st.selectbox("Choose Regression Model",
                                    ["Linear Regression", "SVM Regression",
                                     "KNN Regression", "Decision Tree Regression",
                                     "Random Forest Regression"])

        new_value = st.number_input(f"Input a new value for {x_col} to predict {y_col}", value=0.0)

        if st.button("Perform Regression"):
            X = df[[x_col]]
            y = df[y_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_choice in ["SVM Regression", "KNN Regression"]:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                new_value_scaled = scaler.transform(np.array([[new_value]]))
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                new_value_scaled = np.array([[new_value]])

            if model_choice == "Linear Regression":
                model = LinearRegression()

            elif model_choice == "SVM Regression":
                model = SVR(kernel='rbf')

            elif model_choice == "KNN Regression":
                model = KNeighborsRegressor(n_neighbors=5)

            elif model_choice == "Decision Tree Regression":
                model = DecisionTreeRegressor()

            elif model_choice == "Random Forest Regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            score = model.score(X_test_scaled, y_test)
            st.write(f"R-squared (Accuracy): {score}")
            prediction = model.predict(new_value_scaled)
            st.write(f"Predicted {y_col} for {x_col} = {new_value}: {prediction[0]}")

            plt.figure(figsize=(8, 6))
            plt.scatter(X_test, y_test, color="blue", label="Actual Data")

            if model_choice in ["Linear Regression", "SVM Regression"]:
                plt.plot(X_test, y_pred, color="red", label="Regression Line")
            else:
                plt.scatter(X_test, y_pred, color="green", label="Predicted Data")


            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{model_choice} between {x_col} and {y_col}")
            plt.legend()
            st.pyplot(plt)
