from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
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
from bson import ObjectId

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

# Enable CORS for React Frontend (Port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
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

def serialize_doc(doc):
    """Converts MongoDB ObjectId to string for JSON compatibility."""
    doc["_id"] = str(doc["_id"])
    if "createdAt" in doc and isinstance(doc["createdAt"], datetime):
        doc["createdAt"] = doc["createdAt"].isoformat()
    if "created_at" in doc and isinstance(doc["created_at"], datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc

# --- AUTH ENDPOINTS ---

@app.post("/api/auth/register")
async def register(user: UserRegister):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
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
    return serialize_doc(new_user)

@app.post("/api/auth/login")
async def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Wrong password")
    
    # The frontend expects a user object with an accessToken
    response_user = serialize_doc(db_user)
    response_user["accessToken"] = "simulated_token_for_dev"
    return response_user

# --- POST ENDPOINTS ---

@app.post("/api/posts")
async def create_post(
    userId: str = Form(...),
    desc: str = Form(""),
    file: UploadFile = File(...)
):
    """Creates a post after verifying image isn't AI generated."""
    user = users_collection.find_one({"_id": ObjectId(userId)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create temp file for AI Detection
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Zero AI Content Verification
        print(f"🔍 Verifying content for user: {user['username']}...")
        client = Client(HF_SPACE, hf_token=HF_TOKEN)
        result = client.predict(img=handle_file(tmp_path), api_name="/predict")
        
        # Example result: {'ai': 0.02, 'hum': 0.98}
        ai_score = result.get('ai', 0)
        hum_score = result.get('hum', 1)

        if ai_score > hum_score:
            raise HTTPException(status_code=406, detail="AI Content Detected. Only human creators allowed.")

        # 2. Upload to Cloudinary
        with open(tmp_path, "rb") as f:
            upload_data = {'upload_preset': CLOUDINARY_PRESET}
            cloud_res = requests.post(CLOUDINARY_UPLOAD_URL, files={'file': f}, data=upload_data)
            
        img_url = cloud_res.json().get("secure_url")
        if not img_url:
            raise HTTPException(status_code=500, detail="Failed to host image")

        # 3. Save Post to MongoDB
        new_post = {
            "userId": userId,
            "username": user["username"],
            "desc": desc,
            "img": img_url,
            "likes": [],
            "created_at": datetime.now(),
            "ai_verification_score": ai_score
        }
        posts_collection.insert_one(new_post)
        return serialize_doc(new_post)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/api/posts/timeline/all")
async def get_timeline():
    """Fetches all human-verified posts for the feed."""
    posts = list(posts_collection.find().sort("created_at", -1))
    return [serialize_doc(p) for p in posts]

@app.put("/api/posts/{post_id}/like")
async def like_post(post_id: str, userId: str = Form(...)):
    post = posts_collection.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if userId in post.get("likes", []):
        posts_collection.update_one({"_id": ObjectId(post_id)}, {"$pull": {"likes": userId}})
        return "Post unliked"
    else:
        posts_collection.update_one({"_id": ObjectId(post_id)}, {"$push": {"likes": userId}})
        return "Post liked"

# --- USER ENDPOINTS ---

@app.get("/api/users")
async def get_user(userId: str = None, username: str = None):
    query = {}
    if userId: query["_id"] = ObjectId(userId)
    elif username: query["username"] = username
    
    user = users_collection.find_one(query)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return serialize_doc(user)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)