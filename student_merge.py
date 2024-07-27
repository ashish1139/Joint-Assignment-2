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
df2 = pd.read_csv(r"C:\Users\ASUS\Documents\Python\student-por.csv", sep=";", dtype={"G1": int, "G2": int, "G3": int})

# Merge two datasets on the specified columns
common_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
df3 = pd.merge(df1, df2, on=common_columns)

# Print the number of rows in the merged DataFrame
print(len(df3))
print(df3.info())
print(df3.head())
print(df3.columns)

# Define the numerical columns of interest with suffixes
numerical_columns = ['Medu', 'Fedu', 'Dalc_x', 'Walc_x', 'G1_x', 'G2_x', 'G3_x']

# Check if all numerical columns exist in df3
missing_columns = [col for col in numerical_columns if col not in df3.columns]
if missing_columns:
    print(f"\nMissing columns: {missing_columns}")
else:
    # Compute the correlation matrix for the selected columns
    corr_matrix = df3[numerical_columns].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

# Additional statistical analyses
# Distribution of final grades (G3_x)
plt.figure(figsize=(12, 6))
sns.histplot(df3['G3_x'], kde=True, bins=20)
plt.title('Distribution of Final Grades (G3_x)')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.show()

# Boxplot of final grades (G3_x) by school
plt.figure(figsize=(12, 6))
sns.boxplot(x='school', y='G3_x', data=df3)
plt.title('Final Grades (G3_x) by School')
plt.xlabel('School')
plt.ylabel('Grade')
plt.show()
