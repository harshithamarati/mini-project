import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import streamlit as st

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

# Load model and scaler
model, scaler = load_and_train_model()

# Streamlit app
def main():

    st.title("University Admission Predictor")

    # Explanation about University Rating in expander
    with st.expander("Click here to learn about University Rating"):
        st.markdown("""
        ### What is University Rating?
        The "University Rating" is a numerical value representing the reputation or ranking of the university to which an applicant is applying. 
        This rating is often used as a feature in predictive models to help estimate the likelihood of an applicant being admitted based on 
        the quality or prestige of the institution.

        #### Purpose of University Rating:
        - **Institution Reputation**: Higher-rated universities may have more competitive admissions processes.
        - **Quality of Education**: Indicates the perceived quality of education and resources available at the university.
        - **Applicant's Strategy**: Applicants might apply to a mix of universities with different ratings to balance their chances of admission.

        #### Typical Rating Scale:
        - **1**: Low-rated university
        - **2**: Average-rated university
        - **3**: Good university
        - **4**: Very good university
        - **5**: Top-rated university
        """)

    st.markdown("### Please input your details below:")

    # Collect user input
    GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
    TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
    University_Rating = st.number_input("University Rating", min_value=1, max_value=5, step=1)
    SOP = st.number_input("SOP", min_value=0.0, max_value=5.0, step=0.1)
    LOR = st.number_input("LOR", min_value=0.0, max_value=5.0, step=0.1)
    CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
    Research = st.number_input("Research (0 or 1)", min_value=0, max_value=1, step=1)

    # Predict button
    if st.button("Predict"):
        try:
            features = [GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]
            features_array = np.array(features).reshape(1, -1)

            # Scale the input features
            features_scaled = scaler.transform(features_array)

            # Make a prediction
            prediction = model.predict(features_scaled)[0]

            # Center the image display
            col1, col2, col3 = st.columns([1, 3, 1])

            with col2:
                # Display result
                if prediction < 0.5:
                    st.image("nochance.png", caption="You don't have a chance.", width=250)
                else:
                    st.image("chance.jpg", caption="You have a chance.", width=250)
        except Exception as e:
            st.error(f"There was an error with your input: {e}")

if __name__ == "__main__":
    main()
