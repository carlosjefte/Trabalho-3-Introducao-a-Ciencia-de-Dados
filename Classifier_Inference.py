import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model

# Load the dataset
file_path = 'Crânios Egípcios Formatado.csv'
df_cleaned = pd.read_csv(file_path)

# Load the trained TensorFlow model
model = load_model('tensorflow_cranios_classifier.h5')

# Function to classify skull measurements
def classify_skull(measurements):
    """
    Classifies the skull based on input measurements.
    
    Parameters:
    measurements (list or np.array): The skull measurements
    
    Returns:
    str: The predicted era for the skull
    """
    # Ensure the input is a numpy array and reshape for a single prediction
    measurements = np.array(measurements, dtype=np.float32).reshape(1, -1)  # Convert to float32
    
    # Make a prediction using the TensorFlow model (predict returns probabilities)
    prediction_prob = model.predict(measurements)
    prediction_class = np.argmax(prediction_prob, axis=1)  # Get the class with the highest probability
    
    # Map the prediction back to the era label
    era_map = {
        0: "Pré-dinástico primitivo",
        1: "Pré-dinástico antigo",
        2: "12 e 13 dinastias",
        3: "Período ptolemaico",
        4: "Período romano"
    }
    
    predicted_era = era_map[prediction_class[0]]
    return predicted_era

# Function to randomly select a skull, predict its era, and compare it to the actual label
def random_skull_inference(df, iteration):
    # Randomly select a row from the dataset
    random_index = random.randint(0, len(df) - 1)
    skull_data = df.iloc[random_index]
    
    # Extract the measurements (all the X1-X4 columns)
    measurements = skull_data.drop(['label']).values
    
    # Get the real era label
    real_era = skull_data['label']
    
    # Classify the skull based on its measurements
    predicted_era = classify_skull(measurements)
    
    # Display the result
    print(f"Iteration {iteration}")
    print(f"Randomly selected skull measurements: {measurements}")
    print(f"Predicted Era: {predicted_era}")
    print(f"Actual Era: {real_era}\n")

# Call the function to randomly select a skull and classify it
n = int(input("Enter the number of skulls you want to classify: "))
for i in range(n):
    random_skull_inference(df_cleaned, i + 1)