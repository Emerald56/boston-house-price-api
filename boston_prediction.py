#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load Boston datasets from CSV
df = pd.read_csv("boston.csv")

# Rename all columns from my .csv files for clearer understanding
df.rename(columns={
    "CRIM": "CrimeRate",
    "ZN": "ResidentialLandZone",
    "INDUS": "NonRetailLand",
    "CHAS": "CharlesRiverDummy",
    "NOX": "NitricOxideConcentration",
    "RM": "AvgRoomsPerDwelling",
    "AGE": "ProportionOldHomes",
    "DIS": "DistanceToEmployment",
    "RAD": "HighwayAccessIndex",
    "TAX": "PropertyTaxRate",
    "PTRATIO": "PupilTeacherRatio",
    "B": "ProportionBlack",
    "LSTAT": "LowerStatusPopulation",
    "MEDV": "HousePrice"
}, inplace=True)
print ("Columns:", df.columns)

# Inspect the data
print ('Data Info:')
print (df.info())
print ("\nSummary Statistics:")
print (df.describe())
print ("\nMissing Values:")
print (df.isnull().sum())

# correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Split data into features (x) and target (y)
x = df.drop('HousePrice', axis=1)
y = df['HousePrice']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_importance = pd.Series(lin_reg.coef_, index= x_train.columns)

# Make Predictions
y_pred = lin_reg.predict(x_test)

# Evaluate the Linear Regression Model
mse = mean_squared_error (y_test, y_pred)
mae = mean_absolute_error (y_test, y_pred)
cv_scores = cross_val_score(LinearRegression(), x, y, cv=5, scoring= "r2")
r2 = r2_score (y_test, y_pred)

print (f"\nLinear Regression Model Evaluation")
print (f"Mean Squared Error (MSE): {mse: .2f}")
print ("MAE:", mae)
print (f"R\u00B2 Score: {r2:2f}")
print ("Cross-Validation R\u00B2:", cv_scores.mean())

# Save the model
joblib.dump(lin_reg, 'lin_reg_model.pkl')

# Compare Actual vs Predicted Values
comparison = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred})
print ("\nActual vs Predicted Values (first 10 rows):")
print ( comparison.head(10))

#  Train a Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index= x_train.columns)

# Make Predictions
y_pred_rf = rf.predict(x_test)

# Evaluate the Random Forest Regressor Model
mse = mean_squared_error (y_test, y_pred_rf)
mae = mean_absolute_error (y_test, y_pred_rf)
cv_scores = cross_val_score(RandomForestRegressor(), x, y, cv=5, scoring= "r2")
r2 = r2_score (y_test, y_pred_rf)

print ("\nRandom Forest Results")
print ("MSE", mean_squared_error(y_test, y_pred_rf))
print ("MAE:", mae)
print ("R²:", r2_score(y_test, y_pred_rf))
print ("Cross-Validation R\u00B2:", cv_scores.mean())
print ()

# Save the model
joblib.dump(rf, 'rf_reg_model.pkl')

# Compare Actual vs Predicted Values
comparison = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred_rf})
print ("\nActual vs Predicted Values (first 10 rows):")
print ( comparison.head(10))

# Initialize and Train a Gradient Boosting Regressor Model
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(x_train, y_train)
gb_importance = pd.Series(gb.feature_importances_, index= x_train.columns)

# Make Predictions
y_pred_gb = gb.predict(x_test)

# Evaluate the Gradient Boosting Regressor Model
mse = mean_squared_error (y_test, y_pred_gb)
mae = mean_absolute_error (y_test, y_pred_gb)
cv_scores = cross_val_score(RandomForestRegressor(), x, y, cv=5, scoring= "r2")
r2 = r2_score (y_test, y_pred_gb)

print ("\nGradient Boosting Results.")
print ("MSE:", mean_squared_error(y_test, y_pred_gb))
print ("MAE:", mae)
print ("R²:", r2_score (y_test, y_pred_gb))
print ("Cross-Validation R\u00B2:", cv_scores.mean())

# Save the model
joblib.dump(gb, 'gb_reg_model.pkl')


# Compare Actual vs Predicted Values
comparison = pd.DataFrame({'Actual' : y_test, 'Predicted': y_pred_gb})
print ("\nActual vs Predicted Values (first 10 rows):")
print ( comparison.head(10))


# Actual vs Predicted Plots
def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.show()

plot_actual_vs_predicted(y_test, y_pred, "Linear Regression")
plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
plot_actual_vs_predicted(y_test, y_pred_gb, "Gradient Boosting Regressor")