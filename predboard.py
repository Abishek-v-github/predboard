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
from sklearn.preprocessing import StandardScaler,LabelEncoder

st.title("PredictBoard")

upfile = st.file_uploader("Upload CSV", type=["csv"])

if upfile is not None:

    df = pd.read_csv(upfile)

    st.write("Preview :", df.head())

    st.sidebar.header("Handle Null Values")

    drop_columns = st.sidebar.multiselect("Select Columns to Drop if They Contain Nulls",options=df.columns.tolist())

    if drop_columns:
        df = df.drop(columns=drop_columns)

    fill_columns = st.sidebar.multiselect("Select Columns to Fill Nulls",options=df.columns.tolist())

    fill_strategy = st.sidebar.selectbox("Choose Fill Strategy for Selected Columns",["None", "Fill with Mean", "Fill with Median", "Fill with Mode"])

    if fill_columns and fill_strategy != "None":
        if fill_strategy == "Fill with Mean":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mean())
        elif fill_strategy == "Fill with Median":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].median())
        elif fill_strategy == "Fill with Mode":
            df[fill_columns] = df[fill_columns].fillna(df[fill_columns].mode().iloc[0])


    label_encode_columns = st.sidebar.multiselect(
        "Select Columns to Apply Label Encoding",
        options=df.select_dtypes(include=['object']).columns.tolist())

    if label_encode_columns:
        label_encoder = LabelEncoder()
        for col in label_encode_columns:
            df[col] = label_encoder.fit_transform(df[col])

    st.write("Updated Data :", df)

    if df.empty:
        st.error("Please upload another csv")
    else:
        column_options = df.columns.tolist()
        x_col = st.selectbox("Select X (Independent Variable)[numeric]", column_options)
        y_col = st.selectbox("Select Y (Dependent Variable)[numeric]", column_options)

        modelchoice = st.selectbox("Choose Regression Model",
                                    ["Linear Regression", "SVM Regression",
                                     "KNN Regression", "Decision Tree Regression",
                                     "Random Forest Regression"])

        newval = st.number_input(f"Input a new value for {x_col} to predict {y_col}", value=0.0)

        if st.button("Perform Regression"):
            X = df[[x_col]]
            y = df[y_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if modelchoice in ["SVM Regression", "KNN Regression"]:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                newvalscaled = scaler.transform(np.array([[newval]]))
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                newvalscaled = np.array([[newval]])

            if modelchoice == "Linear Regression":
                model = LinearRegression()

            elif modelchoice == "SVM Regression":
                model = SVR(kernel='rbf')

            elif modelchoice == "KNN Regression":
                model = KNeighborsRegressor(n_neighbors=5)

            elif modelchoice == "Decision Tree Regression":
                model = DecisionTreeRegressor()

            elif modelchoice == "Random Forest Regression":
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            score = model.score(X_test_scaled, y_test)
            st.write(f"R-squared (Accuracy): {score}")
            prediction = model.predict(newvalscaled)
            st.write(f"Predicted {y_col} for {x_col} = {newval}: {prediction[0]}")

            plt.figure(figsize=(8, 6))
            plt.scatter(X_test, y_test, color="blue", label="Actual Data")

            if modelchoice in ["Linear Regression", "SVM Regression"]:
                plt.plot(X_test, y_pred, color="red", label="Regression Line")
            else:
                plt.scatter(X_test, y_pred, color="green", label="Predicted Data")

            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{modelchoice} between {x_col} and {y_col}")
            plt.legend()
            st.pyplot(plt)
