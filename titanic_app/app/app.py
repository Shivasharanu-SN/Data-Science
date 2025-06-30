import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessors
model = joblib.load('../model/model.pkl')
encoder_sex = joblib.load('../model/encoder_sex.pkl')
encoder_embarked = joblib.load('../model/encoder_embarked.pkl')
encoder_deck = joblib.load('../model/encoder_deck.pkl')
constants = joblib.load('../model/constants.pkl')

# Extract constants
age_median = constants['age_median']
feature_order = constants['feature_order']

st.title("Titanic Survival Prediction")

st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0, step=1.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "U"])

# user_input = {
#     'PassengerId': 892,
#     'Pclass': 3,
#     'Name': 'Kelly, Mr. James',
#     'Sex': 'male',
#     'Age': 34.5,
#     'SibSp': 0,
#     'Parch': 0,
#     'Ticket': 330911,
#     'Fare': 7.8292,
#     'Cabin': '',
#     'Embarked': 'Q',
# }

if st.button("Predict Survival"):
    # Build input dict
    user_input = {
        "Pclass": pclass,
        "Age": age,
        'SibSp': sibsp,
        'Parch': parch,
        "Fare": fare,
        "Sex": sex,
        "Embarked": embarked,
        "Deck": deck
    }


    # Convert to Data Frame
    df = pd.DataFrame([user_input])

    # df['Deck'] = df['Cabin'].str[0].fillna('U')

    # df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Fill missing Age
    df['Age'] = df['Age'].fillna(age_median)

    # Encode categorical columns using saved encoders
    encoded_sex = pd.DataFrame(encoder_sex.transform(df[['Sex']]), columns=encoder_sex.get_feature_names_out(['Sex']))
    encoded_embarked = pd.DataFrame(encoder_embarked.transform(df[['Embarked']]), columns=encoder_embarked.get_feature_names_out(['Embarked']))
    encoded_deck = pd.DataFrame(encoder_deck.transform(df[['Deck']]), columns=encoder_deck.get_feature_names_out(['Deck']))

    # Select numerical features
    numeric = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].reset_index(drop=True)

    # Combine all
    X = pd.concat([numeric, encoded_sex, encoded_embarked, encoded_deck], axis=1)

    # Ensure column order matches training
    X = X[feature_order]

    # Predict
    prediction = model.predict(X)[0]
    label = " Survived!" if prediction == 1 else " Did not survive"

    # Display
    st.success(f"Prediction: {label}")
