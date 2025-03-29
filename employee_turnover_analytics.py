import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## 1. Perform data quality checks
# Load dataset
df = pd.read_csv('HR_comma_sep.csv')

# Check for missing values in each column
missing_values = df.isnull().sum()

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Missing values summary table
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage
})
print(missing_data)

## 2. EDA
## 2.1 Draw a heatmap of the correlation matrix
# Select numerical columns
numerical_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                 'average_montly_hours', 'time_spend_company', 'Work_accident',
                 'promotion_last_5years']

# Calculate correlation matrix
corr_matrix = df[numerical_cols + ['left']].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

## 2.2 Draw the distribution plot of:
## a) Employee Satisfaction (use column satisfaction_level)
## b) Employee Evaluation (use column last_evaluation)
## c) Employee Average Monthly Hours (use column average_montly_hours)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(data=df, x='satisfaction_level', hue='left', element='step', ax=axes[0], bins=30, kde=True)
axes[0].set_title('Distribution of Employee Satisfaction Levels')
axes[0].set_xlabel('Satisfaction Level (0-1)')
axes[0].set_ylabel('Count')

sns.histplot(data=df, x='last_evaluation', hue='left', element='step', ax=axes[1], bins=30, kde=True)
axes[1].set_title('Distribution of Employee Evaluation Scores')
axes[1].set_xlabel('Evaluation Score (0-1)')
axes[1].set_ylabel('Count')

sns.histplot(data=df, x='average_montly_hours', hue='left', element='step', ax=axes[2], bins=30, kde=True)
axes[2].set_title('Distribution of Average Monthly Hours')
axes[2].set_xlabel('Average Monthly Hours')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()

## 2.3 Draw the bar plot of the employee project count of both employees who left and stayed in the organization
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='number_project', hue='left')
plt.title('Employee Project Count Distribution by Turnover Status')
plt.xlabel('Number of Projects')
plt.ylabel('Employee Count')
plt.legend(title='Left Company', labels=['No', 'Yes'])
plt.show()

## 3. Perform clustering of employees who left based on their satisfaction and evaluation.
## a) Choose columns satisfaction_level, last_evaluation, and left.
## b) Do K-means clustering of employees who left the company into 3 clusters?
## c) Based on the satisfaction and evaluation factors, give your thoughts on the employee clusters.

# Select employees who left
left_employees = df[df['left'] == 1][['satisfaction_level', 'last_evaluation']]

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
left_employees['cluster'] = kmeans.fit_predict(left_employees)

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=left_employees, x='satisfaction_level', y='last_evaluation', hue='cluster', palette='viridis')
plt.title("K-Means Clustering of Employees Who Left")
plt.xlabel("Satisfaction Level")
plt.ylabel("Last Evaluation")
plt.legend(title="Cluster")
plt.show()

# Calculate cluster means
cluster_summary = left_employees.groupby('cluster').mean()
print(cluster_summary)

# Count employees in each cluster
cluster_counts = left_employees['cluster'].value_counts()
print("\nEmployee count per cluster:")
print(cluster_counts)