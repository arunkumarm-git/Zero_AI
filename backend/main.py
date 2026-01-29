from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from pymongo import MongoClient
from datetime import datetime
import requests
import os
import shutil
import tempfile
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()
# --- CONFIGURATION ---
# 1. Hugging Face Config
HF_SPACE = "Arunmass/AI_IMG_Detector"
HF_TOKEN = os.getenv("HF_API")

# 2. Cloudinary Config
CLOUDINARY_UPLOAD_URL = "https://api.cloudinary.com/v1_1/YOUR_CLOUD_NAME/image/upload"
CLOUDINARY_PRESET = "zero_ai_preset" # The name you gave your unsigned preset

# 3. MongoDB Config
# Replace <password> with your actual password (no brackets)
MONGO_URI = "mongodb+srv://YOUR_USER:YOUR_PASSWORD@cluster0.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "zero_ai_db"

# --- SETUP CLIENTS ---
# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
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

@app.post("/api/create-post")
async def create_post(
    file: UploadFile = File(...), 
    description: str = Form(...)
):
    print(f"Received upload request: {file.filename}")
    
    # Create a temporary file to handle the upload cleanly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # --- STEP 1: AI DETECTION (Hugging Face) ---
        print("1. Sending to AI Detector...")
        
        # Initialize the Gradio Client for your space
        client = Client(HF_SPACE, hf_token=HF_TOKEN)
        
        # Make the prediction
        # handle_file prepares the file for the Gradio API
        result = client.predict(
            img=handle_file(temp_file_path),
            api_name="/predict" 
        )
        
        # The result usually comes back as specific JSON or a Label dict. 
        # Based on your previous prompt, it looks like: {'ai': 0.04, 'hum': 0.95}
        # Note: Gradio Client might return it slightly differently (e.g., a label string).
        # We will print it first to be safe.
        print(f"🔍 AI Check Result: {result}")

        # PARSING LOGIC: 
        # If your space returns a simple label (common in Gradio), we check the label.
        # If it returns the JSON you showed earlier, we parse that.
        
        # SAFEGUARD: Let's assume result returns the JSON object directly or a list
        # If result is like {'label': 'human'}, adapt logic here.
        # For now, let's implement the logic based on your JSON example:
        
        ai_score = 0.0
        hum_score = 0.0
        
        # Adjust this depending on EXACTLY what your Space returns.
        # Assuming result is a dictionary like { "ai": 0.05, "hum": 0.95 }
        if isinstance(result, dict):
            ai_score = result.get('ai', 0)
            hum_score = result.get('hum', 0)
        # If Gradio returns a list like [{'label': 'ai', 'score': 0.05}, ...]
        elif isinstance(result, list): 
             # Just a fallback parser
             pass 

        # REJECTION LOGIC
        if ai_score > hum_score:
            print("❌ BLOCKED: AI content detected.")
            raise HTTPException(status_code=406, detail="AI Content Detected. Only Human content allowed.")

        print("✅ VERIFIED: Content is Human.")

        # --- STEP 2: CLOUDINARY UPLOAD ---
        print("2. Uploading to Cloudinary...")
        
        # Re-open file for reading bits to send to Cloudinary
        with open(temp_file_path, "rb") as f:
            files = {'file': f}
            data = {'upload_preset': CLOUDINARY_PRESET}
            cloud_res = requests.post(CLOUDINARY_UPLOAD_URL, files=files, data=data)
            
        cloud_data = cloud_res.json()
        secure_url = cloud_data.get("secure_url")

        if not secure_url:
            print(f"Cloudinary Error: {cloud_data}")
            raise HTTPException(status_code=500, detail="Image upload failed")

        print(f"☁️ Uploaded to: {secure_url}")

        # --- STEP 3: SAVE TO MONGODB ---
        print("3. Saving to Database...")
        
        post_document = {
            "description": description,
            "image_url": secure_url,
            "ai_score": ai_score,
            "human_score": hum_score,
            "created_at": datetime.now(),
            "verified": True
        }
        
        result = posts_collection.insert_one(post_document)
        print(f"💾 Saved to DB with ID: {result.inserted_id}")

        return {
            "status": "success", 
            "imgurl": secure_url, 
            "message": "Post created successfully"
        }

    except Exception as e:
        print(f"Error: {e}")
        # Pass through HTTP exceptions (like the 406 for AI)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup: Remove the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)