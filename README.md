
 **Boston House Price API**

A machine learning project that predicts house price in the Boston area using multiple regression models. The project demonstrates end-to-end ML workflow, data processing and model training to API deployment with Flask.

**PROJECT OVERVIEW******

This project was built to showcase real-world machine learning deployment. It allows users to send input data (housing features) via an API and receive predicted house prices in response.

The models trained include:

-     Linear Regression
-     Random Forest Regressor
-     Gradient Boosting Regressor

Cross-validation and model comparison were performed to identify the most accurate predictor.

** FEATURES **

1. Trained on the Boston Housing dataset.

2. Supports multiple models with easy switching:
 
-      Linear Regression

-      Random Forest Regressor
 
-         Gradient Boosting Regressor
 
3.   REST API endpoints built with Flask:
 
-       /predict → predict with one model
 
-       /compare → compare predictions across all models
 
4.  Includes cross-validation & evaluation metrics (MSE, MAE, R²).
 
5.    API tested with Postman.


 **Tech Stack**

-     Python (pandas, scikit-learn, joblib, numpy)

-     Flask for API

-     Postman for testing

-   Deployment to Railway

**How to Run**

1. Clone Repo: git clone https://github.com/Emerald56/boston-house-price-api.git
cd boston-house-price-prediction

2. Install requirements: pip install -r requirements.txt

3. Run Flask Server: python app.py

4. Test Endpoint with Postman or Curl

**API Usage**

> Single Model Prediction:

POST/predict
```
{
"features": [0.1, 18.0, 6.5, 0, 0.7, 6.0, 65, 4.0, 1, 300, 15, 390, 5],
"model": "rf"
}

```
Response 
```
{
"model_used": "Random Forest",
"prediction": 23.8
}
```

> Compare All Models


POST /compare
```
{
"features": [0.1, 18.0, 6.5, 0, 0.7, 6.0, 65, 4.0, 1, 300, 15, 390, 5]
}
```

Response 

```
{
"all_predictions": {
"Linear Regression": 21.5,
"Random Forest": 23.8,
"Gradient Boosting": 22.9
}
}
```
**Real-World Impact**

-     Helps homebuyers estimate fair prices.

-     Assists realtors in setting data-driven listings.

-     Provides a foundation for real estate analytics platforms.

