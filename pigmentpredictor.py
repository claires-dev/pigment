import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# Load the data
data = pd.read_csv(r"C:\Users\dirhf260\Downloads\Color Study .csv")
X = data[['WhitePigment', 'BlackPigment', 'RedPigment', 'YellowPigment']]
y = data[['L', 'a', 'b']]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train separate models for L, a, and b
model_L = LinearRegression().fit(X_train, y_train['L'])
model_a = LinearRegression().fit(X_train, y_train['a'])
model_b = LinearRegression().fit(X_train, y_train['b'])

# Extract coefficients and intercepts
coefficients = {
    "L": (model_L.intercept_, model_L.coef_),
    "a": (model_a.intercept_, model_a.coef_),
    "b": (model_b.intercept_, model_b.coef_),
}

# Define the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    target_L = data.get('L')
    target_a = data.get('a')
    target_b = data.get('b')

    if target_L is None or target_a is None or target_b is None:
        return jsonify({"error": "Missing target values"}), 400

    targets = [target_L, target_a, target_b]

    # Define the objective function
    def objective_function(pigments):
        white, black, red, yellow = pigments
        eq_L = coefficients["L"][0] + sum(coefficients["L"][1] * [white, black, red, yellow]) - target_L
        eq_a = coefficients["a"][0] + sum(coefficients["a"][1] * [white, black, red, yellow]) - target_a
        eq_b = coefficients["b"][0] + sum(coefficients["b"][1] * [white, black, red, yellow]) - target_b
        eq_sum = white + black + red + yellow - 1
        return eq_L**2 + eq_a**2 + eq_b**2 + eq_sum**2

    # Set bounds and initial guess
    bounds = [(0, 1)] * 4
    initial_guess = [0.25, 0.25, 0.25, 0.25]

    # Minimize the objective function
    result = minimize(objective_function, initial_guess, bounds=bounds)
    print("Predictions:", predictions)
    if not result.success:
        return jsonify({"error": "Optimization failed"}), 500

    white, black, red, yellow = result.x
    return jsonify({
        "WhitePigment": white,
        "BlackPigment": black,
        "RedPigment": red,
        "YellowPigment": yellow
    })

if __name__ == '__main__':
    app.run(debug=True)


