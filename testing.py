import requests
import json

# Replace with your local or public Ngrok URL if needed
API_URL = "https://leading-remarkably-hare.ngrok-free.app/predict"

# Sample input data
payload = {
    "area_sqft": 1361.25,
    "location": "park view villas",
    "property_type": "house",
    "bedrooms": 3,
    "province_name": "punjab"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction Result:")
        print("Predicted Price:", result.get('final_ensemble_price'))
        print("Investment Advice:", result.get('investment_advice'))
        if result.get('note'):
            print("Note:", result['note'])

        recommendations = result.get('recommendations', [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"📍 Location: {rec['location']}, Price: {rec['price']}, Area: {rec['area_sqft']} sqft, Price/Sqft: {rec['price_per_sqft']}")
        else:
            print("No alternative recommendations found.")
    else:
        print("❌ Error:", response.status_code, response.text)

except Exception as e:
    print("❌ Exception occurred while making the request:", str(e))
