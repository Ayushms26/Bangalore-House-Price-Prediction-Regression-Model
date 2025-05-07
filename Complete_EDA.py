

# Load cleaned data
data = pd.read_csv('cleaned_bangalore_house_data.csv')

# Distribution of price
plt.figure(figsize=(8, 5))
sns.histplot(data['price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price (Lakhs)')
plt.ylabel('Frequency')
plt.show()

# Boxplot to check outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['total_sqft'])
plt.title('Boxplot of Total Square Feet')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# and new feachures
import numpy as np
from sklearn.cluster import KMeans

# Load cleaned data
data = pd.read_csv('cleaned_bangalore_house_data.csv')

# Area per BHK ratio
data['area_per_bhk'] = data['total_sqft'] / data['BHK']

# Apply K-Means clustering on location
kmeans = KMeans(n_clusters=10, random_state=42)
data['location_cluster'] = kmeans.fit_predict(data[['location']])

# Save feature-engineered data
data.to_csv('feature_engineered_bangalore_house_data.csv', index=False)
