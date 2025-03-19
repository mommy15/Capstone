import pymongo
import os
from dotenv import load_dotenv

# Load MongoDB URI from .env file
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["rent_tracker"]
payments_collection = db["payments"]

# Fetch data
data = list(payments_collection.find({}, {"_id": 0}))

if data:
    print("✅ Data Found in MongoDB!")
    print("First record:", data[0])  # Show one sample record
else:
    print("❌ No Data Found in MongoDB!")






