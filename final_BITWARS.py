import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import requests
import json
import apikey

vif_data = pd.DataFrame()


# Chart explanations (appear above the charts, providing useful info)
chart_explanations = {
    "Histogram": "A histogram displays the distribution of a numerical variable, showing how frequently each range of values appears.",
    "Boxplot": "A boxplot helps visualize the distribution of data and highlights outliers using quartiles.",
    "Scatter Plot": "A scatter plot shows relationships between two numerical variables, useful for identifying correlations.",
    "Line Chart": "A line chart displays trends over time or any continuous variable.",
    "Bar Chart": "A bar chart compares different categories using rectangular bars.",
    "Correlation Heatmap": "A heatmap visualizes correlations between numerical variables, with colors indicating strength of relationships.",
    "Distribution Detection": "This feature identifies the type of distribution (e.g., Normal, Skewed) for a selected numerical column."
}


# Streamlit UI
st.title("📊 Illustra : Universal Data Visualization & Machine Learning Tool")
st.write("Upload a CSV file, visualize data, and train basic machine learning models.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_summary = df.describe().to_string()
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Null values in your dataset")
    st.write(df.isna().sum())

    # Handling missing values, filling w mean median mode
    fill_method = st.selectbox("Select a method to fill missing values", ["None", "Mean", "Median", "Mode"])
    if fill_method != "None":
        for column in df.select_dtypes(include=[np.number]).columns:
            if fill_method == "Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif fill_method == "Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif fill_method == "Mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
        st.success(f"Missing values have been replaced using {fill_method}.")
        st.write(df.isna().sum())

    # Multicollinearity Detection with Column Selection and Confirmation
    st.write("### Multicollinearity Detection & Column Dropping")


    def calculate_vif(data):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = data.columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        return vif_data


    numerical_df = df.select_dtypes(include=['number']).dropna()
    vif_df = calculate_vif(numerical_df)
    st.dataframe(vif_df)

    # Allow user to select columns to drop 
    high_vif_cols = st.multiselect("Select columns to drop to reduce multicollinearity",
                                   vif_df[vif_df['VIF'] > 5]['Feature'].tolist())

    # Confirm button to finalize column removal
    if st.button("Confirm Column Selection"):
        if high_vif_cols:
            df = df.drop(columns=high_vif_cols)
            st.success(f"Dropped columns: {', '.join(high_vif_cols)}")

            # Recalculate multicollinearity after dropping columns
            numerical_df = df.select_dtypes(include=['number']).dropna()
            vif_df = calculate_vif(numerical_df)
            st.write("### Updated Multicollinearity Table")
            st.dataframe(vif_df)
        else:
            st.warning("No columns selected for removal.")

    # One-Hot Encoding -
    st.write("### One-Hot Encoding")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_cols = st.multiselect("Select categorical columns to one-hot encode", categorical_cols)
        if st.button("Apply Encoding") and selected_cols:
            df = pd.get_dummies(df, columns=selected_cols, drop_first=True)
            st.success(f"One-Hot Encoding applied to {', '.join(selected_cols)}.")
            st.write("Updated Data Preview:")
            st.dataframe(df.head())
    else:
        st.write("No categorical columns found for one-hot encoding.")


    #Chart selection
    chart_type = st.selectbox("Select a chart type", list(chart_explanations.keys()))
    st.write(f"**About {chart_type}:** {chart_explanations[chart_type]}")

    if chart_type == "Histogram":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.histogram(df, x=column)
        st.plotly_chart(fig)

    elif chart_type == "Boxplot":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.box(df, y=column)
        st.plotly_chart(fig)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis", df.select_dtypes(include=['number']).columns)
        y_col = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns)
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Line Chart":
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns)
        fig = px.line(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Bar Chart":
        x_col = st.selectbox("Select a categorical column", df.select_dtypes(include=['object']).columns)
        y_col = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif chart_type == "Distribution Detection":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)

        # Compute skewness and kurtosis
        skewness = stats.skew(df[column].dropna())
        kurtosis = stats.kurtosis(df[column].dropna())

        # Check normality using Shapiro-Wilk test
        shapiro_test = stats.shapiro(df[column].dropna())
        p_value = shapiro_test.pvalue

        # Determine distribution type
        if p_value > 0.05:
            distribution_type = "Normal Distribution"
        elif skewness > 1 or skewness < -1:
            distribution_type = "Highly Skewed Distribution"
        else:
            distribution_type = "Moderately Skewed Distribution"

        # Display results
        st.write(f"**Distribution Analysis for '{column}':**")
        st.write(f"- **Skewness:** {skewness:.2f}")
        st.write(f"- **Kurtosis:** {kurtosis:.2f}")
        st.write(f"- **Shapiro-Wilk Test p-value:** {p_value:.4f}")
        st.write(f"- **Identified Distribution Type:** {distribution_type}")

        # Plot histogram with KDE
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # deepseek AI integration
    st.write("### Illustra AI")
    model_id = "deepseek/deepseek-r1-distill-llama-70b:free"
    user_input = st.text_area("Enter text for Illustra AI analysis")

    API_KEY = apikey.API_KEY #done to protect api key from unwanted access 
    # Define the OpenRouter API URL
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    if st.button("Analyze with Illustra AI"):
        if not API_KEY:
            st.error("API Key is missing. Please provide a valid OpenRouter API key.")
        elif not user_input:
            st.error("Please enter a message.")
        else:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                    "model": "deepseek/deepseek-r1-distill-llama-70b:free",  # Corrected model ID
                    "messages": [{"role": "system",
                                  "content": "you are a data scientist looking to analyze dataset\n" + csv_summary},
                                 {"role": "user", "content": user_input}]
                }
            # Send request
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

            # Display response
            if response.status_code == 200:
                response_data = response.json()
                reply = response_data["choices"][0]["message"]["content"]
                st.write("🤖 **Illustra AI Response:**")
                st.write(reply)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
    # Machine Learning Models
    st.write("## Train a Machine Learning Model")

    # User selects features to include in the model
    all_features = list(df.columns)
    target_column = st.selectbox("Select Target Variable", df.columns)
    input_features = st.multiselect("Select Input Features", [col for col in all_features if col != target_column])

    if not input_features:
        st.warning("Please select at least one input feature to proceed.")
        st.stop()

    # Select Model Type
    model_type = st.selectbox("Select a Model",
                              ["Linear Regression", "Logistic Regression", "Decision Tree Regressor",
                               "Decision Tree Classifier", "SVM Regressor", "SVM Classifier"])

    # Initialize session state #so it doesn't refresh again and erases everything
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.selected_features = None
        st.session_state.categorical_cols = None

    if st.button("Train Model"):
        X = df[input_features]  # Use only selected features
        y = df[target_column]

        # Convert categorical variables into numerical values
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store selected features & categorical column info
        st.session_state.selected_features = input_features
        st.session_state.categorical_cols = categorical_cols

        # Select Model
        model = None
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_type == "SVM Regressor":
            model = SVR()
        elif model_type == "SVM Classifier":
            model = SVC()

        # Train Model
        model.fit(X_train, y_train)

        # Save model in session state
        st.session_state.model = model

        # Predict on test data
        y_pred = model.predict(X_test)

        # Display Performance Metrics
        if model_type in ["Linear Regression", "Decision Tree Regressor", "SVM Regressor"]:
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
        elif model_type in ["Logistic Regression", "Decision Tree Classifier", "SVM Classifier"]:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        st.success("Model trained successfully! Now you can make predictions below.")

    #--------------------------- Prediction Feature ----------------------------------------------------
    if st.session_state.model:
        st.write("## 🔮 Predict the Target Variable")
        st.write("Enter values for the selected features:")

        user_inputs = {}
        for col in st.session_state.selected_features:
            user_inputs[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))

        # Convert user inputs to DataFrame
        user_df = pd.DataFrame([user_inputs])

        # Ensure categorical columns match the trained model (for one-hot encoding)
        if st.session_state.categorical_cols is not None and not st.session_state.categorical_cols.empty:
            user_df = pd.get_dummies(user_df, columns=st.session_state.categorical_cols, drop_first=True)

            # Add missing columns if any (due to encoding)
            missing_cols = set(st.session_state.selected_features) - set(user_df.columns)
            for col in missing_cols:
                user_df[col] = 0  # Assign 0 to missing categorical columns

            # Ensure column order matches the trained model
            user_df = user_df[st.session_state.selected_features]

        # Make Prediction
        prediction = st.session_state.model.predict(user_df)

        # Display Prediction
        st.write("## 🎯 Predicted Target Variable:")
        st.success(f"Predicted Value: {prediction[0]:.4f}")
