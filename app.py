from flask import Flask, request, jsonify
import joblib
import google.generativeai as genai
import os

app = Flask(__name__)

# Load model (adjust path if deployed on Render with mounted disk)
model = joblib.load("xgboost_crop_yield_model.pkl")

# Configure Gemini API using environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Soil suitability mapping
soil_suitability = {
    "Wheat": ["Loamy", "Clayey"],
    "Rice": ["Clayey", "Loamy"],
    "Maize": ["Loamy", "Sandy"],
    "Sugarcane": ["Clayey", "Loamy"],
    "Cotton": ["Sandy", "Loamy"],
    "Pulses": ["Sandy", "Loamy"],
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Collect input
        state = data["state"]
        crop = data["crop"]
        soil_type = data["soil_type"]
        season = data["season"]
        rainfall_category = data["rainfall_category"]
        area = float(data["area"])

        # Example encoded data (replace with actual encoding logic)
        new_data = [[
            1,  # Replace with actual encoding for state
            2,  # Replace with actual encoding for crop
            3,  # Replace with actual encoding for season
            4,  # Replace with actual encoding for rainfall_category
            area
        ]]

        # Predict yield
        predicted_yield = model.predict(new_data)[0]
        total_yield = predicted_yield * area

        # Soil Suitability Check
        if crop in soil_suitability and soil_type not in soil_suitability[crop]:
            soil_warning = f"⚠️ {soil_type} is not ideal for {crop}. Recommended: {', '.join(soil_suitability[crop])}"
        else:
            soil_warning = f"✅ {soil_type} is ideal for growing {crop}."

        # Create prompt for Gemini
        prompt = f"""
        Generate a detailed farming report based on the following data:
        - State: {state}
        - Crop: {crop}
        - Soil Type: {soil_type}
        - Season: {season}
        - Rainfall Category: {rainfall_category}
        - Land Area: {area:.2f} hectares
        - Predicted Yield per hectare: {predicted_yield:.2f} tons
        - Estimated Total Yield: {total_yield:.2f} tons

        Soil Comment: {soil_warning}

        Include:
        1. Farming suggestions
        2. Fertilizer recommendations
        3. Pest and disease control
        4. Water and irrigation management
        5. Best agricultural practices
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "predicted_yield_per_hectare": predicted_yield,
            "total_yield": total_yield,
            "report": response.text
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
