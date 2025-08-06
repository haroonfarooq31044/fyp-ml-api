import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
df = pd.read_csv("DataSet/Property_with_Feature_Engineering.csv")
df.dropna(subset=['area_sqft', 'location', 'property_type', 'bedrooms', 'province_name', 'price'], inplace=True)

# Features and target
features = ['area_sqft', 'location', 'property_type', 'bedrooms', 'province_name']
X = df[features].copy()
y = df['price'].copy()

# Encode categorical columns
encoders = {}
for col in ['location', 'property_type', 'province_name']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str).str.lower())
    encoders[col] = le

# Scale area
scaler = MinMaxScaler()
X['area_sqft'] = scaler.fit_transform(X[['area_sqft']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/xgb_land_price_model.joblib')
joblib.dump({'encoders': encoders, 'scaler': scaler}, 'model/xgb_preprocessing.joblib')

print("✅ Model training complete and saved.")
