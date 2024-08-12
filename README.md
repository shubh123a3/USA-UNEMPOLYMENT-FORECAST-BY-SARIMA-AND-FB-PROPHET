
# USA Unemployment Rate Forecast


https://github.com/user-attachments/assets/2ff01d1e-b3dd-4701-8d92-a0a9b300faae



This project focuses on forecasting the unemployment rate in the USA using time series analysis techniques. The primary models used are **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** and **Facebook Prophet**.

## Project Overview

The objective was to create a reliable model for predicting the unemployment rate in the USA based on historical data. Both SARIMA and Facebook Prophet models were implemented for this purpose. 

### Methods Used
- **SARIMA**: A statistical model that extends ARIMA by including seasonal components.
- **Facebook Prophet**: A forecasting tool developed by Facebook, which handles seasonality, holidays, and trend changes well.

### Dataset
The data used in this project was the historical unemployment rate in the USA, which was sourced from reliable government and financial databases. The data is stored in the `unemployment_data.csv` file.

### Key Files
- **`Usa Unemplyment Prediction.ipynb`**: The Jupyter notebook containing the code and analysis for model building, evaluation, and forecasting.
- **`fbProfet.pkl`**: The serialized Facebook Prophet model.
- **`fbProfet_predict_model.pkl`**: The model predictions from Facebook Prophet.
- **`helper.py`**: Contains helper functions for data preprocessing and visualization.
- **`app.py`**: A script to deploy the forecasting model.


## Requirements
To run this project, you will need the following Python libraries:
- pandas
- numpy
- matplotlib
- statsmodels
- pmdarima
- fbprophet

These can be installed using the `requirements.txt` file.

## Conclusion
This project demonstrates the challenges in time series forecasting, especially when dealing with economic indicators like unemployment rates. Future improvements could include experimenting with other models or incorporating additional data sources to improve accuracy.

## How to Run
1. Clone the repository: 
   ```
   git clone https://github.com/shubh123a3/USA-UNEMPOLYMENT-FORECAST-BY-SARIMA-AND-FB-PROPHET.git
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook for analysis or the `app.py` script for deployment.

---
### Detailed Project Overview

#### 1. **Introduction**
   - The project aims to forecast the unemployment rate in the USA using time series analysis. The models employed are SARIMA (Seasonal AutoRegressive Integrated Moving Average) and Facebook Prophet, both well-suited for handling seasonal data patterns.

#### 2. **Data Collection**
   - Historical unemployment rate data was sourced from reliable databases. The data includes monthly records of the unemployment rate in the USA over several years. This dataset was cleaned and preprocessed to ensure consistency and accuracy before modeling.

#### 3. **Exploratory Data Analysis (EDA)**
   - Before model building, the data was visualized to understand trends, seasonality, and potential outliers. EDA helped in identifying the underlying patterns and provided insights into the seasonality of the unemployment rates.

#### 4. **SARIMA Model**
   - **Model Selection**: SARIMA was selected due to its capability to handle both trend and seasonality. The model's order parameters (p, d, q) and seasonal parameters (P, D, Q, s) were determined using ACF and PACF plots.
   - **Model Fitting**: The SARIMA model was fitted to the training data, with an emphasis on tuning parameters to optimize the fit.
   - **Prediction**: The SARIMA model was used to forecast the unemployment rate. However, the predictions showed limitations in capturing future trends accurately.

#### 5. **Facebook Prophet Model**
   - **Model Selection**: Facebook Prophet was chosen for its flexibility in handling seasonal data, holidays, and abrupt trend changes. It also allows for automatic fitting and forecasting.
   - **Model Fitting**: The model was trained on the unemployment data, with customizations to include yearly seasonality and holiday effects.
   - **Prediction**: The Prophet model generated forecasts that were more reliable than SARIMA but still faced challenges in accurately predicting the unemployment rate.

#### 6. **Model Evaluation**
   - **Performance Metrics**: The models were evaluated using metrics such as RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). SARIMA’s performance was suboptimal, with high error rates. Prophet performed better but still lacked precision in long-term forecasts.
   - **Comparison**: Both models were compared to assess their strengths and weaknesses. While Prophet outperformed SARIMA, neither model provided sufficiently accurate predictions for practical applications.

#### 7. **Conclusion**
   - The project highlighted the complexities of time series forecasting, particularly for economic indicators like unemployment rates. Although both SARIMA and Prophet models were implemented, the results indicated the need for further refinement, potentially by incorporating additional data sources or trying alternative models.

#### 8. **Future Work**
   - Future enhancements could involve experimenting with machine learning-based time series models or integrating external economic factors to improve prediction accuracy. Additionally, fine-tuning the existing models and exploring ensemble approaches could yield better results.

---

