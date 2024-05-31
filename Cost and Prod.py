import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("testdata.csv")


attribute1 = df["EmployeeEngagement"]
attribute2 = df["WorkLife_Balance"]
attribute3 = df["Training_Development"]

correlation_matrix = df[["EmployeeEngagement","WorkLife_Balance","Training_Development"]].corr()


plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.show()
X = df[['EmployeeEngagement','WorkLife_Balance','Training_Development']]
y = df['CostProd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the regularization strength (alpha)
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred = ridge_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.title("Scatter Plot of Predicted vs. Actual Values")
plt.xlabel("Actual Talent")
plt.ylabel("Predicted Talent")
plt.show()

# Specify the column names you want to create pie charts for
columns_to_plot = ['EmployeeEngagement','WorkLife_Balance','Training_Development']  # Replace with your desired column names

# Create pie charts for the specified columns
plt.figure(figsize=(50, 30))

# Loop through the specified columns and create pie charts
for i, column in enumerate(columns_to_plot, start=1):
    plt.subplot(1, len(columns_to_plot), i)
    counts = df[column].value_counts().sort_index()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=360)
    plt.title(f'{column} Ratings')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
