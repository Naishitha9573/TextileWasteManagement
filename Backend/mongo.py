import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

MONGODB_URL = os.getenv("MONGODB_URL", "").strip()
client = None


def init_mongo():
    global client
    if not MONGODB_URL:
        client = None
        return None
    if client is None:
        try:
            client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=2000)
            client.server_info()
        except Exception as exc:
            client = None
            print(f"[MONGO] Connection attempt failed: {exc}", flush=True)
    return client


def get_mongo_db():
    if not MONGODB_URL:
        return None
    if client is None:
        init_mongo()
    if client is None:
        return None
    try:
        return client.get_default_database()
    except Exception:
        return None


def close_mongo():
    global client
    if client:
        try:
            client.close()
        except Exception:
            pass
        client = None

