import pymongo
import bcrypt
import jwt
import datetime
import os
from dotenv import load_dotenv



# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")

# MongoDB Connection
client = pymongo.MongoClient(MONGO_URI)
db = client["rent_tracker"]
users_collection = db["users"]
payments_collection = db["payments"]

# User Registration (Landlord or Tenant)
def register_user(email, password, role="tenant", landlord_email=None):
    if users_collection.find_one({"email": email}):
        return {"error": "User already exists"}
    
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    
    user_data = {"email": email, "password": hashed_password, "role": role}
    if role == "tenant" and landlord_email:
        user_data["landlord_email"] = landlord_email  # Link tenant to landlord

    users_collection.insert_one(user_data)
    return {"message": f"{role.capitalize()} registered successfully"}

# User Login
def login_user(email, password):
    user = users_collection.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return {"error": "Invalid credentials"}
    
    token = jwt.encode(
        {"email": email, "role": user["role"], "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
        JWT_SECRET,
        algorithm="HS256"
    )
    return {"token": token, "role": user["role"]}

# Add Rent Payment
def add_rent_payment(tenant_email, landlord_email, monthly_income, rent_amount, payment_delay, previous_late_payments, status="Paid"):
    """Stores rent payment data in MongoDB"""
    payment_data = {
        "tenant": tenant_email,
        "landlord": landlord_email,
        "monthly_income": monthly_income,
        "rent_amount": rent_amount,
        "payment_delay": payment_delay,
        "previous_late_payments": previous_late_payments,
        "status": status,  # "Paid" or "Late"
        "date": str(datetime.date.today())
    }
    payments_collection.insert_one(payment_data)
    return {"message": "Rent payment added successfully"}

# Get Rent Payments
def get_rent_payments(email, role):
    if role == "tenant":
        payments = list(payments_collection.find({"tenant": email}, {"_id": 0}))
    elif role == "landlord":
        payments = list(payments_collection.find({"landlord": email}, {"_id": 0}))
    else:
        return {"error": "Invalid role"}

    return payments if payments else {"message": "No payments found"}


