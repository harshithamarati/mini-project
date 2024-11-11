import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
from flask import Flask, request, render_template

# Function to train the model
def train_model(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Check model accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    print("Training R^2 score:", train_accuracy)
    print("Test R^2 score:", test_accuracy)

    # Save the scaler and model to file
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('admission_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, scaler

# Load the data and train the model
def load_and_train_model():
    data = pd.read_csv('admission_data.csv')
    model, scaler = train_model(data)
    return model, scaler

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model, scaler = load_and_train_model()

# Home route
@app.route('/')
def home():
    return render_template('user.html')

# Prediction route
@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Get form data and convert to float
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        # Scale the input features
        features_scaled = scaler.transform(features_array)

        # Make a prediction
        prediction = model.predict(features_scaled)[0]

        # Debugging: print input and output
        print("Input values:", features)
        print("Scaled values:", features_scaled)
        print("Prediction output:", prediction)

        # Return result based on prediction output
        if prediction < 0.5:
            return render_template('nochance.html', prediction_text='You don\'t have a chance.')
        else:
            return render_template('chance.html', prediction_text='You have a chance.')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('error.html', prediction_text='There was an error with your input.')

if __name__ == "__main__":
    app.run(debug=True)
