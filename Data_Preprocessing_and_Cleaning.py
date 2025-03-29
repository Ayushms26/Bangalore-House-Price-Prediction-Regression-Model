import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('Bengaluru_House_Data.csv')

# Drop unnecessary columns
data.drop(columns=['society', 'area_type', 'availability'], inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Convert 'size' column to numerical (extract BHK count)
data['BHK'] = data['size'].apply(lambda x: int(str(x).split(' ')[0]))

# Convert 'total_sqft' column to numerical
def convert_sqft(sqft):
    try:
        return float(sqft)
    except:
        temp = sqft.split('-')
        if len(temp) == 2:
            return (float(temp[0]) + float(temp[1])) / 2
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data.dropna(subset=['total_sqft'], inplace=True)

# Encode 'location' column
label_encoder = LabelEncoder()
data['location'] = label_encoder.fit_transform(data['location'])

# Feature scaling
scaler = StandardScaler()
data[['total_sqft', 'bath', 'balcony']] = scaler.fit_transform(data[['total_sqft', 'bath', 'balcony']])

# Save cleaned data
data.to_csv('cleaned_bangalore_house_data.csv', index=False)
