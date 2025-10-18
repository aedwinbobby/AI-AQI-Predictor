pip install pip
C:Users/VICTUS/AppData/Local/Programs/Python/Python314/python.exe -m pip install gradio


import gradio as gr
import pickle
import numpy as np

# -----------------------------
# Load your trained model
# -----------------------------
model_path = "model/aqi_model.pkl"  # Adjust if saved elsewhere
with open(model_path, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Air Quality Category function
# -----------------------------
def get_air_quality(aqi):
    if aqi <= 50:
        return "Good 🌿"
    elif aqi <= 100:
        return "Moderate 😊"
    elif aqi <= 200:
        return "Unhealthy for Sensitive Groups 😷"
    elif aqi <= 300:
        return "Unhealthy ☠️"
    elif aqi <= 400:
        return "Very Unhealthy 💀"
    else:
        return "Hazardous 🚫"

# -----------------------------
# Prediction function
# -----------------------------
def predict_gases(aqi_value):
    # Ensure correct input format
    aqi_value = float(aqi_value)
    X = np.array([[aqi_value]])

    # Predict all gas values
    predicted = model.predict(X)[0]  # returns list/array
    gases = ["CO (ppm)", "NO2 (µg/m³)", "SO2 (µg/m³)", "O3 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

    results = {gas: round(val, 3) for gas, val in zip(gases, predicted)}

    # Determine air quality category
    air_quality = get_air_quality(aqi_value)

    return results, air_quality

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align:center;'>🌍 AI AQI Predictor</h1>")
    gr.Markdown("Enter AQI value to get predicted gas concentrations and air quality level.")

    aqi_input = gr.Number(label="Enter AQI Value", value=100)
    predict_btn = gr.Button("Predict")

    gas_output = gr.JSON(label="Predicted Gas Concentrations")
    quality_output = gr.Textbox(label="Air Quality Level")

    predict_btn.click(predict_gases, inputs=aqi_input, outputs=[gas_output, quality_output])

# -----------------------------
# Launch app
# -----------------------------
if __name__ == "__main__":
    demo.launch()
