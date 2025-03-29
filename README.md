# Bangalore House Price Prediction – Regression Model  

## Project Overview  
This project aims to predict house prices in Bangalore using a **Linear Regression model**. The dataset is preprocessed by handling missing values, removing outliers using the **IQR method**, and applying **feature engineering** techniques such as **location clustering and area-per-BHK ratio**. The model achieves an **R² score of 0.86** and is deployed as a **Flask web application**, allowing users to input house details and obtain real-time price predictions.  

## Key Features  
- **Linear Regression Model** trained with optimized feature selection.  
- **Achieved R² Score of 0.86** after feature engineering.  
- **Data Cleaning & Preprocessing:**  
  - Missing values handled  
  - Outliers removed using the **IQR method**  
- **Exploratory Data Analysis (EDA):**  
  - Distribution of house prices  
  - Correlation heatmaps  
  - Boxplots for outlier detection  
- **Feature Engineering:**  
  - **Location Clustering:** Applied **K-Means** to group locations.  
  - **Area per BHK Ratio:** Captures space efficiency for better predictions.  
- **Model Deployment using Flask:**  
  - User inputs house details to get **real-time price predictions**.  
  - Web interface built with **HTML and Flask**.  

## Model Training  
To train the model on a new dataset, run:  
```
python model.py
```
This script performs training and saves the trained model as `house_price_model.pkl`.  

## Results  
- The model achieves an **R² score of 0.86**.  
- Price predictions improve after applying **feature engineering**.  
- The application provides instant price estimates based on user inputs.  

## Future Enhancements  
- Implement **Random Forest Regression** for better performance.  
- Add **Google Maps integration** to enhance location-based price predictions.  
- Develop an interactive **dashboard** for better visualization.  
