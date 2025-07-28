from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model (ensure the path is correct)
model = pickle.load(open('model/carbon_model.pkl', 'rb'))

# All 42 features required by the model
full_features = ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 'Waste Bag Weekly Count','How Long TV PC Daily Hour', 
                 'How Many New Clothes Monthly', 'How Long Internet Daily Hour','Recycling_Paper', 'Recycling_Plastic', 'Recycling_Metal', 
                 'Recycling_Glass','Cooking_Oven', 'Cooking_Microwave', 'Cooking_Airfryer', 'Cooking_Stove', 'Cooking_Grill','Body Type_obese',
                 'Body Type_overweight', 'Body Type_underweight','Sex_male', 'Diet_pescatarian', 'Diet_vegan', 'Diet_vegetarian',
                 'How Often Shower_less frequently', 'How Often Shower_more frequently', 'How Often Shower_twice a day',
                 'Heating Energy Source_electricity', 'Heating Energy Source_natural gas', 'Heating Energy Source_wood','Vehicle Type_electric',
                 'Vehicle Type_hybrid', 'Vehicle Type_lpg', 'Vehicle Type_petrol','Social Activity_often', 'Social Activity_sometimes',
                 'Frequency of Traveling by Air_never', 'Frequency of Traveling by Air_rarely', 'Frequency of Traveling by Air_very frequently',
                 'Waste Bag Size_large', 'Waste Bag Size_medium', 'Waste Bag Size_small','Energy efficiency_Sometimes', 'Energy efficiency_Yes']

# The 6 numeric input features from the form
numeric_features = ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 'Waste Bag Weekly Count','How Long TV PC Daily Hour', 
                    'How Many New Clothes Monthly', 'How Long Internet Daily Hour']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from form
    form_data = request.form

    try:
        # Create a dictionary for all 42 features
        input_features = {}

        # Add numeric features from the form
        for feature in numeric_features:
            input_features[feature] = float(form_data.get(feature))

        # Fill the rest of the features with 0
        for feature in full_features:
            if feature not in input_features:
                input_features[feature] = 0

        # Convert the feature dict to ordered list matching model input
        input_array = [input_features[feature] for feature in full_features]

        # Predict
        prediction = model.predict([input_array])[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Estimated Carbon Emission: {prediction} kg CO₂/month")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
