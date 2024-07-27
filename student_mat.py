# Importing packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the student-mat data
df1 = pd.read_csv(r"C:\Users\ASUS\Documents\Python\student-mat.csv", sep=";", dtype={"G1": int, "G2": int, "G3": int})
print(df1.info())
print(df1.head())

# Data Cleaning and Preprocessing
# Check for missing values
print(df1.isnull().sum())

# Descriptive statistics for numerical variables
numerical_stats = df1.describe()
print("Descriptive Statistics for Numerical Variables:")
print(numerical_stats)

# Descriptive statistics for categorical variables
categorical_columns = df1.select_dtypes(include=['object']).columns
categorical_stats = {}
for column in categorical_columns:
    categorical_stats[column] = df1[column].value_counts()

print("\nDescriptive Statistics for Categorical Variables:")
for column, stats in categorical_stats.items():
    print(f"\n{column}:\n{stats}")

# Detailed descriptive statistics using the 'describe' method for the entire dataframe, including categorical columns
detailed_stats = df1.describe(include='all')
print("\nDetailed Descriptive Statistics for All Variables:")
print(detailed_stats)


# Exploratory Data Analysis

# Convert categorical variables to numeric using Label Encoding
label_encoders = {}
for column in df1.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df1[column] = le.fit_transform(df1[column])
    label_encoders[column] = le

# Plot distributions of the final grade (G3)
plt.figure(figsize=(10, 6))
sns.histplot(df1['G3'], kde=True)
plt.title('Distribution of Final Grades (G3)')
plt.show()

# Pairplot to see relationships
sns.pairplot(df1[['G3', 'G1', 'G2', 'age', 'studytime', 'failures', 'absences']])
plt.show()

# Compute the correlation matrix
corr_matrix = df1.corr()

# Filter the correlation matrix to keep only correlations >= 0.5
filtered_corr_matrix = corr_matrix.where(corr_matrix.abs() >= 0.5)

# Plot the filtered correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(filtered_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=0.5, vmax=1.0, mask=filtered_corr_matrix.isnull())
plt.title('Correlation Matrix (corr >= 0.5)')
plt.show()

# Select the specific columns
columns_of_interest = ['Medu', 'Fedu', 'Dalc', 'Walc', 'G1', 'G2', 'G3']
subset_df1 = df1[columns_of_interest]

# Compute the correlation matrix for the selected columns
corr_matrix1 = subset_df1.corr()

# Print the correlation matrix
print("Correlation Matrix for selected columns:")
print(corr_matrix1)


# Feature Selection
# Selecting features based on correlation with G3
correlation = df1.corr()
relevant_features = correlation['G3'].abs().sort_values(ascending=False).index
print(relevant_features)

# Select top N features (example: top 10 features)
selected_features = relevant_features[1:11]
X = df1[selected_features]
y = df1['G3']

# Data Splitting and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Calculate the difference between actual and predicted values
difference = y_test - y_pred

# Create a color map based on the difference
colors = np.where(difference > 0, 'red', 'green')

# Plotting predicted vs actual with different colors
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c=colors, alpha=0.7)
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades')

# Create custom legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Underestimated')
green_patch = mpatches.Patch(color='green', label='Overestimated')
plt.legend(handles=[red_patch, green_patch])

plt.show()

