import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

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
st.bar_chart(df['Age'].value_counts().sort_index())

# Salary distribution
st.write("Estimated Salary Distribution")
st.bar_chart(df['EstimatedSalary'].value_counts().sort_index())

# Correlation heatmap
st.write("Correlation Heatmap")
st.dataframe(df.corr())

# Model Training
st.subheader("Logistic Regression Model")

# Feature Scaling
X = df.drop('Purchased', axis=1)
y = df['Purchased']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(pd.DataFrame(cm, index=['Not Purchased', 'Purchased'], columns=['Predicted Not Purchased', 'Predicted Purchased']))

# ROC-AUC curve
st.write("ROC-AUC Curve")
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_df = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr
})

st.line_chart(roc_df, x='False Positive Rate', y='True Positive Rate')

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)

st.bar_chart(feature_importance.set_index('feature'))

# Prediction
st.subheader("Make a Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 70, 30)
salary = st.slider("Estimated Salary", 15000, 150000, 50000)

gender_encoded = 1 if gender == "Male" else 0
age_salary_interaction = age * salary

if st.button("Predict"):
    input_data = scaler.transform([[gender_encoded, age, salary, age_salary_interaction]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Prediction: {'Will Purchase' if prediction[0] == 1 else 'Will Not Purchase'}")
    st.write(f"Probability of Purchase: {probability:.2f}")

