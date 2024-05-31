import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor  # Use RandomForestRegressor for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv("testdata.csv")
attribute1 = df["Working Conditions"]
attribute2 = df["Relationship between Colleagues"]
attribute3 = df["Job Satisfaction"]
attribute4 = df["Rewards and Awards"]
attribute5 = df["Workload and Support"]

# Calculate the correlation matrix
correlation_matrix = df[["Working Conditions","Relationship between Colleagues","Job Satisfaction","Rewards and Awards","Workload and Support"]].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.show() 
# Define the features (columns) that you want to use for analysis
features = [
    'Working Conditions',
    'Relationship between Colleagues',
    'Job Satisfaction',
    'Company Policies',
    'Rewards and Awards',
    'Workload and Support'
]

# Define the target variable (the column you want to predict)
target = 'Culture_metrics'  # Replace with your target column name


# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot actual vs. predicted ratings (you can use scatter plot or line plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='red', alpha=0.7)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs. Predicted Employee Satisfaction Ratings')
plt.show()
