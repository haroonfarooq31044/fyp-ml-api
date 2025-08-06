from flask import Flask, request, jsonify
import pandas as pd
import joblib
from difflib import get_close_matches
import os

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load('model/xgb_land_price_model.joblib')
objects = joblib.load('model/xgb_preprocessing.joblib')
encoders = objects['encoders']
scaler = objects['scaler']

# Load dataset
df_original = pd.read_csv("DataSet/Property_with_Feature_Engineering.csv")
df_original.dropna(subset=['area_sqft', 'location', 'property_type', 'bedrooms', 'province_name', 'price'], inplace=True)

@app.route('/')
def index():
    return jsonify({"message": "✅ Land Price Prediction API is running"}), 200

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        area = float(data['area_sqft'])
        location = data['location'].strip().lower()
        property_type = data['property_type'].strip().lower()
        bedrooms = int(data['bedrooms'])
        province = data['province_name'].strip().lower()

        note = ""

        # Location fuzzy match
        known_locs = df_original[df_original['province_name'].str.lower() == province]['location'].str.lower().unique().tolist()
        loc_match = get_close_matches(location, known_locs, n=1, cutoff=0.5)
        used_location = loc_match[0] if loc_match else location
        if location != used_location:
            note += f"⚠️ Location '{location}' replaced with '{used_location}'\n"

        # Prepare input
        input_dict = {
            'area_sqft': area,
            'location': used_location,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'province_name': province
        }
        input_df = pd.DataFrame([input_dict])

        # Encode categorical values
        for col in ['location', 'property_type', 'province_name']:
            val = input_df.at[0, col]
            if val not in encoders[col].classes_:
                close = get_close_matches(val, encoders[col].classes_, n=1, cutoff=0.5)
                if close:
                    input_df.at[0, col] = close[0]
                    note += f"⚠️ '{val}' replaced with '{close[0]}' for {col}\n"
                else:
                    return jsonify({'error': f"Invalid value for {col}: {val}"}), 400
            input_df[col] = encoders[col].transform(input_df[col])

        input_df['area_sqft'] = scaler.transform(input_df[['area_sqft']])

        # Predict
        pred_price = float(model.predict(input_df)[0])
        advice = "Profitable Investment ✅" if pred_price <= 10000000 else "Not Profitable ❌"

        recommendations = []
        if "Not Profitable" in advice:
            similar = df_original[
                (df_original['province_name'].str.lower() == province) &
                (df_original['property_type'].str.lower() == property_type) &
                (df_original['bedrooms'] == bedrooms)
            ].copy()

            similar['price_per_sqft'] = similar['price'] / similar['area_sqft']
            cheaper = similar[similar['price'] < pred_price].sort_values(by='price').head(3)

            for _, row in cheaper.iterrows():
                recommendations.append({
                    'location': row['location'],
                    'price': int(row['price']),
                    'area_sqft': int(row['area_sqft']),
                    'price_per_sqft': round(row['price_per_sqft'], 2)
                })

        return jsonify({
            'final_ensemble_price': round(pred_price, 2),
            'investment_advice': advice,
            'note': note.strip(),
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_locations_by_province', methods=['GET'])
def get_locations():
    try:
        province = request.args.get('province', '').lower()
        if not province:
            return jsonify({'error': 'Province is required'}), 400

        filtered_locations = df_original[
            df_original['province_name'].str.lower() == province
        ]['location'].dropna().unique().tolist()

        return jsonify({'locations': sorted(filtered_locations)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
