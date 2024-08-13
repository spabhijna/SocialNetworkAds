import pandas as pd
import numpy as np 
import matplotlib.pylab as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import seaborn as sns
import joblib

# Load Preprocesed data from the CSV File
file_path = '/Volumes/Code/SocialNetworkAds/Data/Preprocessed_Social_Network_Ads.csv'
df = pd.read_csv(file_path)

# Prepare features and target Variable
X = df.drop('Purchased', axis=1)
Y = df['Purchased']

# Spliting Data one for traing and other for testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Initialise and train the logistic model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

# Make Predictions
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test) #for log loss

# Evaluate the Model
print("Confusion Matrix:")
print(confusion_matrix(Y_test,Y_pred))

print("\nClassification Report")
print(classification_report(Y_test,Y_pred))

print("Log Loss")
print(log_loss(Y_test, Y_prob))

# plotting weight of feature (Coefficients of variables)
feature_name = X.columns
coefficients = model.coef_.flatten()

plt.figure(figsize=(12,8))
sns.barplot(x=feature_name,y=coefficients)
plt.xticks(fontsize = 10)
plt.title('Feature Importance (Coefficients) for Logistic Regression')
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.show()

# Save the model to a file
model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")





