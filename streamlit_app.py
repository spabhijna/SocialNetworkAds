

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    return pd.read_csv("https://raw.githubusercontent.com/spabhijna/SocialNetworkAds/main/Data/Preprocessed_Social_Network_Ads.csv")

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
ax.hist(df['Age'], bins=20, edgecolor='black')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Salary distribution
st.write("Estimated Salary Distribution")
fig, ax = plt.subplots()
ax.hist(df['EstimatedSalary'], bins=20, edgecolor='black')
ax.set_xlabel('Estimated Salary')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(df.corr(), cmap='coolwarm')
ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=45, ha='right')
ax.set_yticklabels(df.columns)
plt.colorbar(im)
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
im = ax.imshow(cm, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Not Purchased', 'Purchased'])
ax.set_yticklabels(['Not Purchased', 'Purchased'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center')
plt.colorbar(im)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots()
ax.barh(feature_importance['feature'], feature_importance['importance'])
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importance')
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
