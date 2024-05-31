import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("testdata.csv")

attribute1 = df["Relevance"]
attribute2 = df["Quality"]
attribute3 = df["Mentorship"]
attribute2 = df["Advancement_Opportunities"]
attribute3 = df["Impact_on_Skills"]

correlation_matrix = df[["Relevance","Quality","Mentorship","Advancement_Opportunities","Impact_on_Skills"]].corr()


plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.show()
X = df[['Relevance','Quality','Mentorship','Advancement_Opportunities','Impact_on_Skills']]
y = df['Training']

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
# Specify the column names you want to create pie charts for
columns_to_plot = ['Relevance','Quality','Mentorship','Advancement_Opportunities','Impact_on_Skills']  # Replace with your desired column names

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
