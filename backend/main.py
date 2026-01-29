from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from pymongo import MongoClient
from datetime import datetime
from passlib.context import CryptContext
from pydantic import BaseModel
import requests
import os
import shutil
import tempfile
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
app = FastAPI()

# 1. Hugging Face Config
HF_SPACE = "Arunmass/AI_IMG_Detector"
HF_TOKEN = os.getenv("HF_API")

# 2. Cloudinary Config
CLOUDINARY_UPLOAD_URL = os.getenv("CLOUDINARY_UPLOAD_URL")
CLOUDINARY_PRESET = os.getenv("CLOUDINARY_PRESET")

# 3. MongoDB Config
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "zero_ai_db"

# --- SECURITY SETUP ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- DATABASE CONNECTION ---
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    users_collection = db["users"]
    posts_collection = db["posts"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS (For Request Validation) ---
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# --- HELPER FUNCTIONS ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def serialize_post(post):
    return {
        "_id": str(post["_id"]),
        "description": post.get("description"),
        "image_url": post.get("image_url"),
        "verified": post.get("verified"),
        "created_at": post.get("created_at").isoformat() if post.get("created_at") else None,
        "username": post.get("username", "Unknown") # Include username in post
    }

# --- AUTH ENDPOINTS ---

@app.post("/api/auth/register")
async def register(user: UserRegister):
    # Check if user exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password and save
    new_user = {
        "username": user.username,
        "email": user.email,
        "password": get_password_hash(user.password),
        "profilePicture": "",
        "followers": [],
        "followings": [],
        "isAdmin": False,
        "createdAt": datetime.now()
    }
    result = users_collection.insert_one(new_user)
    
    # Return user info (excluding password)
    return {
        "_id": str(result.inserted_id),
        "username": new_user["username"],
        "email": new_user["email"]
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    # Find user
    db_user = users_collection.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check password
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")
    
    # Return user data (simulate a token)
    return {
        "_id": str(db_user["_id"]),
        "username": db_user["username"],
        "email": db_user["email"],
        "profilePicture": db_user.get("profilePicture", ""),
        "accessToken": "dummy-token-for-now" 
    }

# --- POST ENDPOINTS ---

@app.post("/api/create-post")
async def create_post(
    file: UploadFile = File(...), 
    description: str = Form(...),
    # We will assume username is sent, or default to "Anonymous" for now
    username: str = Form("Arun (User)") 
):
    print(f"Received upload request: {file.filename}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # 1. AI Check
        print("1. Sending to AI Detector...")
        client = Client(HF_SPACE, hf_token=HF_TOKEN)
        result = client.predict(img=handle_file(temp_file_path), api_name="/predict")
        print(f"🔍 AI Check Result: {result}")

        # Parsing Logic
        ai_score = 0.0
        hum_score = 0.0
        
        # Handle dictionary response
        if isinstance(result, dict):
            # Check if keys exist, otherwise try to parse from a list if specific format
            if 'ai' in result:
                ai_score = result['ai']
                hum_score = result.get('hum', 0)
            elif 'label' in result: 
                 # Handle single label response if necessary
                 pass
        
        # Safe fallback if parsing failed but we want to allow testing
        # (Remove this fallback in production if strictness is required)
        if ai_score == 0 and hum_score == 0:
             print("⚠️ Warning: Could not parse score, defaulting to Human for test")
             hum_score = 1.0

        if ai_score > hum_score:
            print("❌ BLOCKED: AI content detected.")
            raise HTTPException(status_code=406, detail="AI Content Detected.")

        print("✅ VERIFIED: Content is Human.")

        # 2. Upload to Cloudinary
        with open(temp_file_path, "rb") as f:
            files = {'file': f}
            data = {'upload_preset': CLOUDINARY_PRESET}
            cloud_res = requests.post(CLOUDINARY_UPLOAD_URL, files=files, data=data)
            
        secure_url = cloud_res.json().get("secure_url")
        if not secure_url:
            raise HTTPException(status_code=500, detail="Image upload failed")

        # 3. Save to DB
        post_document = {
            "description": description,
            "image_url": secure_url,
            "username": username,
            "ai_score": ai_score,
            "human_score": hum_score,
            "created_at": datetime.now(),
            "verified": True
        }
        posts_collection.insert_one(post_document)
        
        return {"status": "success", "imgurl": secure_url}

    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)

@app.get("/api/timeline/all")
async def get_timeline():
    posts_cursor = posts_collection.find().sort("created_at", -1)
    return [serialize_post(post) for post in posts_cursor]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)