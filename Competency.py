import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from the CSV file
file_path = "testdata.csv" # Update with the correct file path
df = pd.read_csv(file_path)
attribute1 = df["HCRI"]
attribute2 = df["LEVEL"]

correlation_matrix = df[["HCRI","LEVEL"]].corr()


plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.show()
X = df[['HCRI']]
y = df['LEVEL']

# Explore and preprocess the data if needed
# Example: Handle missing values, encode categorical variables, etc.

# Define a function to categorize "HCRI" into levels (customize as needed)
def categorize_hcri(hcri_value):
    if hcri_value < 75:
        return 'Level 1'
    elif hcri_value < 85:
        return 'Level 2'
    elif hcri_value < 90:
        return 'Level 3'
    else:
        return 'Level 4'

# Create a new column "Level" based on "HCRI"
df['Level'] = df['HCRI'].apply(categorize_hcri)

# Split the data into features (X) and target (y)
X = df[['HCRI']]
y = df['Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluation metrics for Logistic Regression
classification_rep_logistic = classification_report(y_test, y_pred_logistic)

# Confusion matrix (classification)
confusion_mat = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print("Logistic Regression Metrics:")
print(classification_rep_logistic)

