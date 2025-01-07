import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import fsolve, minimize
import numpy as np

#to connect to flask
from flask import Flask, request, jsonify
import joblib
import requests

# Define tthe Flask app
app = Flask(_name_)
url = "https://abc123.ngrok.io/predict"


# Load the data
data = pd.read_csv(r"C:\Users\dirhf260\Downloads\Color Study .csv")
# Update feature set to include the interaction term
X = data[['WhitePigment', 'BlackPigment', 'RedPigment', 'YellowPigment']]  # Features used for training
y = data[['L', 'a', 'b']]  # Target variables

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#responses
y_L = data['L']
y_a = data['a']
y_b = data['b']

# Assuming X contains predictors (e.g., white and black pigments) and y_L, y_a, y_b are response variables
# Fit regression models for L, a, and b
model_L = LinearRegression().fit(X, y_L)
model_a = LinearRegression().fit(X, y_a)
model_b = LinearRegression().fit(X, y_b)

# Extract intercepts and coefficients
intercept_L = model_L.intercept_
beta_white_L, beta_black_L, beta_red_L, beta_yellow_L = model_L.coef_

intercept_a = model_a.intercept_
beta_white_a, beta_black_a, beta_red_a, beta_yellow_a = model_a.coef_

intercept_b = model_b.intercept_
beta_white_b, beta_black_b, beta_red_b, beta_yellow_b = model_b.coef_



# Define the regression model coefficients
beta_L = [intercept_L, beta_white_L, beta_black_L, beta_red_L, beta_yellow_L]  # Coefficients for L
beta_a = [intercept_a, beta_white_a, beta_black_a, beta_red_L, beta_yellow_L]  # Coefficients for a
beta_b = [intercept_b, beta_white_b, beta_black_b, beta_red_L, beta_yellow_L]  # Coefficients for b
print(beta_L)
print(beta_a)
print(beta_b)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON data
    data = request.get_json()
    target_L = data.get('L')
    target_a = data.get('a')
    target_b = data.get('b')

# Create prediction (example logic)
    target = [target_L, target_a, target_b]
    predictions = model.predict([target])  # Adjust based on your model structure



# Define the objective function as the sum of squared errors
def objective_function(pigments):
    white, black, red, yellow = pigments
    # Calculate the residuals for each equation
    eq1 = beta_L[0] + beta_L[1]*white + beta_L[2]*black + beta_L[3]*red + beta_L[4]*yellow - target_L
    eq2 = beta_a[0] + beta_a[1]*white + beta_a[2]*black + beta_a[3]*red + beta_a[4]*yellow - target_a
    eq3 = beta_b[0] + beta_b[1]*white + beta_b[2]*black + beta_b[3]*red + beta_b[4]*yellow - target_b
    eq4 = white + black + red + yellow - 1  # Constraint that pigments sum to 1
    return eq1**2 + eq2**2 + eq3**2 + eq4**2  # Sum of squared residuals

# Define bounds for each pigment (e.g., >= 0)
bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # No upper limit

# Initial guess for the pigment values
initial_guess = [1.125, 0.07, 1.46, 0.67]

# Minimize the objective function
result = minimize(objective_function, initial_guess, bounds=bounds)

# Get the solution
solution = result.x

white_solution, black_solution, red_solution, yellow_solution = solution
print(f"White pigment concentration: {white_solution}")
print(f"Black pigment concentration: {black_solution}")
print(f"Red pigment concentration: {red_solution}")
print(f"Yellow pigment concentration: {yellow_solution}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

### Attempting to correct for high VIF results
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assuming `X` contains the pigment data and `y_L`, `y_a`, `y_b` are the Lab target values.

# Step 1: Standardize the predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize pigment data

# Step 2: Apply PCA
pca = PCA()  # By default, PCA retains all components
X_pca = pca.fit_transform(X_scaled)

# View explained variance ratios
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)

# Choose number of components (e.g., enough to explain 95% of variance)
n_components = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
print(f"Number of components explaining 95% variance: {n_components}")


