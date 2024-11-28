import streamlit as st
import requests
import pandas as pd
import pickle
import numpy as np

st.title('Song Stream Prediction :musical_note:')

def load_df():
    # Load the song dataset 
    df = pd.read_csv("Spotify_Youtube.csv")  
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df["Duration_s"]=numeric_df["Duration_ms"]/1000
    columnsToBeDropped=['Views','Likes','Comments','Duration_ms']
    numeric_df = numeric_df.drop(columnsToBeDropped,axis=1)
    numeric_df_copy = numeric_df.copy()
    numeric_df = numeric_df.drop(['Stream'], axis=1)
    numeric_df = pd.DataFrame(numeric_df, columns=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo', 'Duration_s'])
    return df, numeric_df

df, numeric_df = load_df()

def get_feature_min_max(df):
    min_max = {}
    for feature in ['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo']:
        min_max[feature] = {
            'min': df[feature].min(),
            'max': df[feature].max()
        }
    return min_max

min_max_values = get_feature_min_max(df)

# Fields for the song stream prediction
fields = {
    "Danceability": {"type": "slider", "required": True, "min": min_max_values["Danceability"]['min'], "max": min_max_values["Danceability"]['max']},
    "Energy": {"type": "slider", "required": True, "min": min_max_values["Energy"]['min'], "max": min_max_values["Energy"]['max']},
    "Loudness": {"type": "slider", "required": True, "min": min_max_values["Loudness"]['min'], "max": min_max_values["Loudness"]['max']},
    "Speechiness": {"type": "slider", "required": True, "min": min_max_values["Speechiness"]['min'], "max": min_max_values["Speechiness"]['max']},
    "Acousticness": {"type": "slider", "required": True, "min": min_max_values["Acousticness"]['min'], "max": min_max_values["Acousticness"]['max']},
    "Liveness": {"type": "slider", "required": True, "min": min_max_values["Liveness"]['min'], "max": min_max_values["Liveness"]['max']},
    "Valence": {"type": "slider", "required": True, "min": min_max_values["Valence"]['min'], "max": min_max_values["Valence"]['max']},
    "Tempo": {"type": "slider", "required": True, "min": min_max_values["Tempo"]['min'], "max": min_max_values["Tempo"]['max']},
    "Duration_s": {"type": "number", "required": True}
}

def check_required_fields(user_data, fields):
    missing_fields = []
    for field, properties in fields.items():
        if properties["required"] and (field not in user_data or user_data[field] is None or user_data[field] == ""):
            missing_fields.append(field)
    return missing_fields

def get_user_input(fields):
    user_data = {}
    num_columns = 2
    columns = st.columns(num_columns)

    for idx, (field, options) in enumerate(fields.items()):
        field_type = options["type"]
        required = options["required"]
        if field_type == "number":
            user_data[field] = columns[idx % num_columns].number_input(
                f"Enter {field}{'*' if required else ''}", 0.0
            )
        if field_type == "slider":
            # Use a slider for the numeric fields
            user_data[field] = columns[idx % num_columns].slider(
                f"Select {field}{'*' if required else ''}",
                min_value=options["min"],
                max_value=options["max"],
                value=(options["min"] + options["max"]) / 2  # Default value set to midpoint
            )

    return user_data

def load_model_and_scaler():
    import joblib
    import os
    from sklearn.preprocessing import StandardScaler
    from model import train
    # Load the trained model and scaler from pickle files
    # model = pickle.load(open("random_forest_model.pkl", "rb"))
    model_path = 'random_forest_model.pkl'  # Update with the correct path to your model file

    # Check if the model file exists
    if os.path.exists(model_path):
        print("Model file exists!")
        model = joblib.load(model_path)
    else:
        
        st.write(f"No model file found!\nTraining Started........\n")
        model = train(numeric_df,df)
    
    scaler=StandardScaler()
    scaler.fit(numeric_df)
    scaler_y = StandardScaler()
    scaler_y.fit(pd.DataFrame(df, columns = ['Stream']))

    
    return model, scaler, scaler_y

# Get the user input


# Process the user input
def process_user_input(user_data, scaler):
    # Convert the user data dictionary to a DataFrame for scaling
    input_data = pd.DataFrame([user_data])
    
    # Scale the input data (excluding 'Duration_s' if you don't want to scale it)
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Function to reverse the scaling on the predicted value
def inverse_scale_prediction(predicted_value, scaler):
    # If the target is scaled, inverse the scaling (assuming the model was trained with scaling)
    # Note: This assumes 'y' (target) was scaled using scaler; adjust according to your model
    return scaler.inverse_transform(predicted_value.reshape(-1,1))

# Handle user input and prediction
user_data = get_user_input(fields)
print(user_data)
print(type(user_data))

# Check for missing fields
missing_fields = check_required_fields(user_data, fields)
if missing_fields:
    st.error(f"The following fields are required: {', '.join(missing_fields)}")
else:
    if st.button("Predict Stream Count"):
        with st.spinner("Processing data..."):
            # Load the model and scaler
            model, scaler, scaler_y = load_model_and_scaler()

            # Process the user data for prediction (scale the features)
            scaled_data = process_user_input(user_data, scaler)
            # Make the prediction
            prediction = model.predict(pd.DataFrame(scaled_data, columns=user_data.keys()))
            prediction =   prediction.reshape(-1, 1)

            # If the target variable was scaled, inverse the scaling
            predicted_stream_count = scaler_y.inverse_transform(prediction)
            
            # Display the predicted stream count
            st.title(f"Predicted Stream Count: {predicted_stream_count[0][0]:.2f}")