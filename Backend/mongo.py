import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/textile_intelligence")
client = None


def init_mongo():
    global client
    if client is None:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Verify connection
        client.server_info()
    return client


def get_mongo_db():
    if client is None:
        init_mongo()
    return client.get_default_database()


def close_mongo():
    if client:
        client.close()
