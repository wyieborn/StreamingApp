import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from difflib import SequenceMatcher, get_close_matches
import unicodedata

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
    "Duration_s": {"type": "number", "required": True},
    "Artist" : {"type": "dropdown", "required": False}
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
            
        elif field_type == "dropdown":
            artist_list = list(map(normalize_string, df['Artist'].unique()))
            # Use selectbox for dropdown field
            closest_artist = get_close_matches("", artist_list, n=1, cutoff=0.0)[0] if artist_list else None
            user_data[field] = columns[idx % num_columns].selectbox(
                f"Select {field}{'*' if required else ''}",
                options=artist_list,
                index=artist_list.index(closest_artist) if closest_artist else 0
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



# Function to normalize strings
def normalize_string(s):
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')

# Define a function for approximate matching with normalization
def find_closest_match(target, df_column, threshold=0.8):
    # Normalize the target and DataFrame values
    normalized_target = normalize_string(target)
    print(normalized_target)
    normalized_values = df_column.apply(normalize_string)
    
    matches = [
        value for value, norm_value in zip(df_column, normalized_values)
        if SequenceMatcher(None, normalized_target, norm_value).ratio() >= threshold
    ]
    return matches

def find_artist(target_string, df):
    # Apply the function to the DataFrame
    threshold = 0.8  # Set the similarity threshold
    matches = find_closest_match(target_string, df['Artist'], threshold)

    if matches:
        print(f"Matches found: {tuple(set(matches))[0]}")
        st.write("found."+ str(tuple(set(matches))[0]))
        return tuple(set(matches))[0]
    else:
        return "No close matches found."
        





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


def artist_wight(data, artist_name):
    # Get top 100 unique artists by streams
    top_artists = (
        data.groupby("Artist")
        .agg({"Stream": "sum"})
        .sort_values(by="Stream", ascending=False)
        .head(100)
        .reset_index()
    )

    # Assign weights based on rank (higher rank, higher weight)
    max_weight = 1.0  # You can set a different scale if needed
    top_artists["Rank"] = range(1, len(top_artists) + 1)
    top_artists["Weight"] = max_weight - (top_artists["Rank"] - 1) / len(top_artists)

    # Print top artists with weights
    
    wt = top_artists[top_artists['Artist']==artist_name]['Weight']
    print("wignt of artist",wt)
    if len(wt)>0:
        return wt.iloc[0]
    
    else :
        return 0.0
    

def adjust_features_with_weight(features, artist_weight):
    factor=0.1
    
    adjusted_features = {}
    for feature, value in features.items():
        if feature in ["Energy", "Danceability"]:  # Features to amplify
            adjusted_features[feature] = value * (1 + factor * artist_weight)
        else:
            adjusted_features[feature] = value
    return adjusted_features

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
            artist_name = user_data['Artist']
            user_data.pop('Artist')
            # Process the user data for prediction (scale the features)
            art_wt = artist_wight(df, artist_name)
            print("artist wight final ", art_wt)
            user_data = adjust_features_with_weight(user_data, art_wt)
            scaled_data = process_user_input(user_data, scaler)
            # Make the prediction
            scaled_df = pd.DataFrame(scaled_data, columns=user_data.keys())
            
            prediction = model.predict(scaled_df)
            prediction =   prediction.reshape(-1, 1)

            # If the target variable was scaled, inverse the scaling
            predicted_stream_count = scaler_y.inverse_transform(prediction)
            pred = f"{predicted_stream_count[0][0]:,.2f}"
            # Display the predicted stream count
            st.title(f"Predicted Stream Count: {pred}")
