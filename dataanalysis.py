# Import all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the data from CSV file named Social_Network_Ads
file_path = "/Volumes/Code/SocialNetworkAds/Data/Social_Network_Ads.csv"
df = pd.read_csv(file_path)

# Display first few cells
print(df.head())

# Basic Understanding of Data
print(f"Dataset info:\n {df.info()}")
print(f"Summary stats:\n {df.describe()}")

# Dropping User ID feature
df = df.drop(df.columns[0], axis=1)

# Encoding 'Gender' Feature
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).astype(int)

# Handling missing values
missing_values = df.isnull().sum()
print(f"Total missing values:\n{missing_values}")

# Handling Outliers
# Show the number of rows before outlier removal
print(f"Number of rows before outlier removal: {len(df)}")

# Define a function to remove outliers based on z-score
def remove_outliers(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

# Remove outliers from 'EstimatedSalary' and 'Age'
df = remove_outliers(df, 'EstimatedSalary')
df = remove_outliers(df, 'Age')

# Show the number of rows after outlier removal
print(f"Number of rows after outlier removal: {len(df)}")

plt.figure(figsize=(15,10))
df.boxplot()
plt.xticks(rotation=90)
plt.title('Box plot of all columns after outlier removal')
plt.show()

# Adding age-salary interaction feature
df['Age_Salary_Interaction'] = df['Age'] * df['EstimatedSalary']

# Define bin edges and labels
bins = [10,20, 30, 40, 50, 60, 70, 80]
labels = ['11-20','21-30', '31-40', '41-50', '51-60', '61-70', '71-80']

# Apply binning
df['Age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Reorder columns to place 'Age_Salary_Interaction' before 'Purchased'
columns = [col for col in df.columns if col not in ['Purchased', 'Age_Salary_Interaction']]
columns = columns + ['Age_Salary_Interaction', 'Purchased']
df = df[columns]

# Plotting binned data
plt.figure(figsize=(10,6))
df['Age_bin'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Age Range')
plt.ylabel('Frequency')
plt.title('Age Distribution by Binned Categories')
plt.show()

print(df)

# Analyzing variables using correlation matrix
# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr() 
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.yticks(rotation=90)
plt.title('Correlation Matrix')
plt.show()

# Plotting Violin Plots
plt.figure(figsize=(14, 6))

# Violin plot for EstimatedSalary by Purchased status
plt.subplot(1,2,1)
sns.violinplot(x='Purchased',y='EstimatedSalary',data = df)
plt.title('Violin plot for EstimatedSalary by Purchased status')


# Violin plot for Age by Purchased status
plt.subplot(1, 2, 2)
sns.violinplot(x='Purchased', y='Age', data=df)
plt.title('Violin Plot of Age by Purchase')

plt.tight_layout()
plt.show()

# Save the preprocessed DataFrame to a CSV file
numeric_df.to_csv("/Volumes/Code/SocialNetworkAds/Data/Preprocessed_Social_Network_Ads.csv", index=False)




