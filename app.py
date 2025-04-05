from flask import Flask, request, jsonify
import joblib
import google.generativeai as genai

app = Flask(__name__)

# Load model
model = joblib.load("xgboost_crop_yield_model.pkl")

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Soil suitability
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

        # Example input format — Replace with actual model feature vector
        new_data = [[
            1,  # Example: state code
            2,  # Example: crop code
            3,  # Example: season code
            4,  # Example: rainfall_category code
            area
        ]]

        predicted_yield = model.predict(new_data)[0]
        total_yield = predicted_yield * area

        # Soil Suitability
        if crop in soil_suitability and soil_type not in soil_suitability[crop]:
            soil_warning = f"{soil_type} is not ideal for {crop}."
        else:
            soil_warning = f"{soil_type} is ideal for {crop}."

        # Create Prompt for Gemini
        prompt = f"""
        Generate a detailed farming report for:
        - State: {state}
        - Crop: {crop}
        - Soil: {soil_type}
        - Season: {season}
        - Rainfall: {rainfall_category}
        - Area: {area} hectares
        - Predicted Yield: {predicted_yield:.2f} tons/hectare
        - Total Yield: {total_yield:.2f} tons

        Soil comment: {soil_warning}

        Include: farming tips, fertilizer advice, pest control, irrigation guidance, and harvesting suggestions.
        """

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "predicted_yield_per_hectare": predicted_yield,
            "total_yield": total_yield,
            "report": response.text,
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
