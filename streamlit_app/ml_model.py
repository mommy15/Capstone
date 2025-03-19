import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pymongo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["rent_tracker"]
payments_collection = db["payments"]

# Ensure "models/" directory exists
os.makedirs("models", exist_ok=True)

# Status Mapping for Multi-Class Classification
status_mapping = {
    "Paid": 0,
    "Late": 1,
    "Due 1 Month": 2,
    "Due 2 Months": 3,
    "Due 3 Months": 4
}
reverse_status_mapping = {v: k for k, v in status_mapping.items()}  # Reverse mapping

# Fetch rent payments from MongoDB
def fetch_rent_data():
    data = list(payments_collection.find({}, {"_id": 0}))  # Get all records except `_id`

    if not data:
        print("❌ MongoDB is empty! Add rent payment records.")
        return None

    # ✅ Convert to DataFrame
    df = pd.DataFrame(data)

    # ✅ Ensure required columns exist
    required_cols = ["monthly_income", "rent_amount", "payment_delay", "previous_late_payments", "status"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing columns in MongoDB: {missing_cols}")
        return None

    # ✅ Remove unwanted columns if they exist
    df = df[required_cols]

    # ✅ Convert to numeric format
    for col in ["monthly_income", "rent_amount", "payment_delay", "previous_late_payments"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # ✅ Drop rows with NaN values
    df.dropna(inplace=True)

    print(f"✅ Successfully loaded {len(df)} records from MongoDB!")
    return df



# Train Model
def train_model():
    df = fetch_rent_data()

    if df is None or df.empty:
        print("❌ No data found in MongoDB. Please add rent payment data first.")
        return

    # ✅ Feature Selection: Now includes `payment_delay`
    X = df[["monthly_income", "rent_amount", "previous_late_payments", "payment_delay"]].copy()
    y = df["status"].map(status_mapping)  # Convert status to numerical labels

    # ✅ Normalize `monthly_income`, `rent_amount`, `payment_delay`
    X["monthly_income"] = np.log1p(X["monthly_income"])  
    X["rent_amount"] = np.log1p(X["rent_amount"])
    X["payment_delay"] = np.log1p(X["payment_delay"] + 1)

    # ✅ Add New Feature: Income-to-Rent Ratio
    X["income_rent_ratio"] = X["monthly_income"] / (X["rent_amount"] + 1)  

    # ✅ Train-Test Split (Balanced Distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # ✅ Train Multi-Class XGBoost Model
    model = xgb.XGBClassifier(
        objective="multi:softmax",  
        num_class=len(status_mapping),  
        n_estimators=100,  
        learning_rate=0.1,  
        max_depth=3,  
        gamma=1.0,  
        min_child_weight=10,  
        reg_lambda=1.5,  
        reg_alpha=1.0,  
        subsample=0.7,  
        colsample_bytree=0.7,  
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # ✅ Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # ✅ Save Model
    model_path = "models/rent_predictor.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved at: {model_path}")

    # Feature Importance
    feature_importance = model.feature_importances_

    # ✅ Print Results in Terminal
    print("\n📌 Model Evaluation Metrics:")
    print(f"✔️ Accuracy: {accuracy:.2f}")
    print(f"✔️ Precision: {precision:.2f}")
    print(f"✔️ Recall: {recall:.2f}")
    print(f"✔️ F1 Score: {f1:.2f}")

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📌 Feature Importance:")
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # ✅ Perform Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    print(f"\n📌 Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Predict Function
def predict_rent_payment(features):
    model_path = "models/rent_predictor.pkl"

    # ✅ Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model file not found! Train the model first.")
        return "Error: Model not found!"

    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return f"Error loading model: {e}"

    # ✅ Apply same transformations as in training
    try:
        features["monthly_income"] = np.log1p(features["monthly_income"])
        features["rent_amount"] = np.log1p(features["rent_amount"])
        features["payment_delay"] = np.log1p(features["payment_delay"] + 1)
        features["income_rent_ratio"] = features["monthly_income"] / (features["rent_amount"] + 1)

        # ✅ Ensure correct feature order
        feature_list = ["monthly_income", "rent_amount", "previous_late_payments", "payment_delay", "income_rent_ratio"]
        input_data = np.array([features[f] for f in feature_list]).reshape(1, -1)

        # ✅ Predict
        prediction = model.predict(input_data)[0]

        return reverse_status_mapping.get(prediction, "Unknown")  # Avoid KeyError
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return f"Error in prediction: {e}"
 
 

  








