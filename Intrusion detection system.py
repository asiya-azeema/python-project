
# 1. Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load the Dataset
# Path to the CSV file (after extraction)
csv_file_path = r"C:\Users\azoos\Downloads\extracted_files\Friday-WorkingHours-Afternoon-DDoS.csv"

# Load the dataset
data = pd.read_csv(csv_file_path)

# 3. Clean Column Names
# Clean the column names by stripping spaces
data.columns = data.columns.str.strip()

# 4. Preview the Data
# Print the first few rows of the dataset to understand the data
print(data.head())

# 5. Plot 1: Distribution of Labels (Normal vs Attack)
sns.countplot(x='Label', data=data, palette='Set2')
plt.title('Distribution of Labels (Normal vs Attack)')
plt.show()

# 6. Plot 2: Distribution of Numerical Features
# Selecting a few numerical features to visualize
numerical_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Flow Bytes/s', 'Flow Packets/s'
]

# Create histograms for these numerical features
data[numerical_features].hist(bins=30, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Selected Numerical Features')
plt.tight_layout()
plt.show()

# 7. Plot 3: Pairwise Relationships
# This will visualize relationships between features for both normal and attack labels
sns.pairplot(data[numerical_features + ['Label']], hue='Label', palette='Set1', diag_kind='kde')
plt.suptitle('Pairwise Relationships between Features', y=1.02)
plt.show()

# 8. Plot 4: Correlation Matrix
# Compute the correlation matrix of numerical features
corr = data[numerical_features].corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 9. Plot 5: Boxplots for Feature Comparison (Normal vs Attack)
# Select one or two features to compare between normal and attack labels
plt.figure(figsize=(12, 6))
sns.boxplot(x='Label', y='Flow Duration', data=data, palette='Set2')
plt.title('Flow Duration Comparison: Normal vs Attack')
plt.show()
