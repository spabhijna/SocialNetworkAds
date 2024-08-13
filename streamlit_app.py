import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Set page title
st.set_page_config(page_title="Social Network Ads Analysis")

# Title
st.title("Social Network Ads Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Preprocessed_Social_Network_Ads.csv")

df = load_data()

# Display raw data
st.subheader("Raw Data")
st.write(df.head())

# Basic statistics
st.subheader("Basic Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Visualizations")

# Age distribution
st.write("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x='Age', kde=True, ax=ax)
st.pyplot(fig)

# Salary distribution
st.write("Estimated Salary Distribution")
fig, ax = plt.subplots()
sns.histplot(data=df, x='EstimatedSalary', kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Model Training
st.subheader("Logistic Regression Model")

X = df.drop('Purchased', axis=1)
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
plt.title('Feature Importance')
st.pyplot(fig)

# Prediction
st.subheader("Make a Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 30)
salary = st.slider("Estimated Salary", 15000, 150000, 50000)

gender_encoded = 1 if gender == "Male" else 0
age_salary_interaction = age * salary

if st.button("Predict"):
    input_data = [[gender_encoded, age, salary, age_salary_interaction]]
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Prediction: {'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'}")
    st.write(f"Probability of Purchase: {probability:.2f}")
