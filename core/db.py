import os
import logging
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import uuid
import datetime

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "docrag")

# Global client
client = None
db = None

def get_db():
    global client, db
    if client is None:
        try:
            client = MongoClient(MONGO_URI)
            db = client[MONGO_DB_NAME]
            
            # Setup indexes
            db.users.create_index("email", unique=True)
            db.documents.create_index([("workspace_id", 1), ("_id", 1)])
            
            logger.info("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return db

# --- USER MANAGEMENT ---

def create_user(email: str, password: str) -> Dict[str, Any]:
    database = get_db()
    users: Collection = database.users
    workspaces: Collection = database.workspaces
    
    if users.find_one({"email": email}):
        raise ValueError("User with this email already exists.")
    
    user_id = str(uuid.uuid4())
    password_hash = generate_password_hash(password)
    
    user_doc = {
        "_id": user_id,
        "email": email,
        "password_hash": password_hash,
        "created_at": datetime.datetime.utcnow()
    }
    users.insert_one(user_doc)
    
    # Auto-create a default workspace for the user
    workspace_id = str(uuid.uuid4())
    workspace_doc = {
        "_id": workspace_id,
        "name": "My Workspace",
        "owner_id": user_id,
        "created_at": datetime.datetime.utcnow()
    }
    workspaces.insert_one(workspace_doc)
    
    return {"user_id": user_id, "email": email, "default_workspace_id": workspace_id}

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    database = get_db()
    user = database.users.find_one({"email": email})
    
    if user and check_password_hash(user["password_hash"], password):
        # Fetch their default workspace
        workspace = database.workspaces.find_one({"owner_id": user["_id"]})
        return {
            "user_id": user["_id"],
            "email": user["email"],
            "workspace_id": workspace["_id"] if workspace else None
        }
    return None

# --- DOCUMENT MANAGEMENT ---

def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    database = get_db()
    return database.documents.find_one({"_id": doc_id})

def list_workspace_documents(workspace_id: str) -> List[Dict[str, Any]]:
    database = get_db()
    docs = list(database.documents.find({"workspace_id": workspace_id}).sort("created_at", -1))
    return docs

def create_document(doc_id: str, workspace_id: str, user_id: str, file_name: str) -> None:
    database = get_db()
    
    doc = {
        "_id": doc_id,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "file_name": file_name,
        "status": "uploaded",
        "qdrant_collection": workspace_id,
        "total_pages": 0,
        "total_chunks": 0,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow()
    }
    database.documents.insert_one(doc)

def update_document_status(doc_id: str, status: str, updates: Dict[str, Any] = None) -> None:
    database = get_db()
    
    set_fields = {"status": status, "updated_at": datetime.datetime.utcnow()}
    if updates:
        set_fields.update(updates)
        
    database.documents.update_one({"_id": doc_id}, {"$set": set_fields})

def delete_document(doc_id: str) -> None:
    database = get_db()
    database.documents.delete_one({"_id": doc_id})

# --- CONVERSATION MANAGEMENT ---

def get_conversation(session_id: str) -> List[Dict[str, Any]]:
    database = get_db()
    conv = database.conversations.find_one({"_id": session_id})
    return conv.get("messages", []) if conv else []

def save_conversation(session_id: str, workspace_id: str, messages: List[Dict[str, Any]]) -> None:
    database = get_db()
    
    database.conversations.update_one(
        {"_id": session_id},
        {
            "$set": {
                "workspace_id": workspace_id,
                "messages": messages,
                "updated_at": datetime.datetime.utcnow()
            },
            "$setOnInsert": {
                "created_at": datetime.datetime.utcnow()
            }
        },
        upsert=True
    )

def clear_conversation(session_id: str) -> None:
    database = get_db()
    database.conversations.delete_one({"_id": session_id})
