from flask import Flask, request, jsonify
from scipy.optimize import minimize
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the Flask app
app = Flask(__name__)

# Load the model (example model creation)
data = pd.read_csv(r"C:\Users\dirhf260\Downloads\Color Study .csv")
X = data[['WhitePigment', 'BlackPigment', 'RedPigment', 'YellowPigment']]
y = data[['L', 'a', 'b']]
model = LinearRegression().fit(X, y)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.get_json()
        target_L = float(data.get('L'))
        target_a = float(data.get('a'))
        target_b = float(data.get('b'))

        # Objective function for optimization
        def objective_function(pigments, target_L, target_a, target_b):
            white, black, red, yellow = pigments
            eq1 = model.coef_[0][0] + model.coef_[0][1]*white + model.coef_[0][2]*black + model.coef_[0][3]*red + model.coef_[0][4]*yellow - target_L
            eq2 = model.coef_[1][0] + model.coef_[1][1]*white + model.coef_[1][2]*black + model.coef_[1][3]*red + model.coef_[1][4]*yellow - target_a
            eq3 = model.coef_[2][0] + model.coef_[2][1]*white + model.coef_[2][2]*black + model.coef_[2][3]*red + model.coef_[2][4]*yellow - target_b
            eq4 = white + black + red + yellow - 1  # Constraint: pigments sum to 1
            return eq1**2 + eq2**2 + eq3**2 + eq4**2

        # Initial guess for pigment values
        initial_guess = [0.25, 0.25, 0.25, 0.25]

        # Minimize the objective function
        bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # Bounds for pigment concentrations
        result = minimize(objective_function, initial_guess, args=(target_L, target_a, target_b), bounds=bounds)

        # Extract the solution
        white_solution, black_solution, red_solution, yellow_solution = result.x

        # Return predictions as JSON
        return jsonify({
            "WhitePigment": white_solution,
            "BlackPigment": black_solution,
            "RedPigment": red_solution,
            "YellowPigment": yellow_solution
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



