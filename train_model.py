import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("health_data.csv")

# Features
X = data[["glucose", "haemoglobin", "cholesterol"]]

# Target
y = data["prediction"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

# Save files
joblib.dump(model, "health_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model trained successfully!")