import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:Naishitha9573@localhost:5432/Textile"
)
MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb://localhost:27017/textile_intelligence"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # Recycling Facility Operator, Sustainability Manager, Textile Manufacturer, Administrator
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    batches = relationship("WasteBatch", back_populates="creator")
    notifications = relationship("Notification", back_populates="user")
class WasteBatch(Base):
    __tablename__ = "waste_batches"
    id = Column(Integer, primary_key=True, index=True)
    fabric_type = Column(String, nullable=False)  # Cotton, Polyester, Wool, Silk, Linen, Denim, Nylon, Rayon, Acrylic, Mixed Fabrics
    source = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)  # in kg
    color = Column(String, nullable=False)
    condition = Column(String, nullable=False)  # Excellent, Good, Fair, Poor, Contaminated
    collection_date = Column(String, nullable=False)
    status = Column(String, default="Registered")  # Registered, Analyzed, Processed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    creator = relationship("User", back_populates="batches")
    analysis = relationship("AnalysisResult", back_populates="batch", uselist=False, cascade="all, delete-orphan")
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("waste_batches.id"), unique=True)
    fabric_texture = Column(String)  # Soft, Coarse, Knit, Woven
    fabric_pattern = Column(String)  # Solid, Striped, Patterned
    fabric_color = Column(String)  # Color name / hex code
    damage_detected = Column(Boolean, default=False)
    contamination_detected = Column(Boolean, default=False)
    
    # Scores
    recyclability_score = Column(Float)
    reuse_score = Column(Float)
    sustainability_score = Column(Float)
    material_recovery_score = Column(Float)
    overall_circularity_score = Column(Float)
    circularity_category = Column(String)  # Excellent Recovery Potential, High, Moderate, Limited, Disposal Recommended
    
    # Recommendations
    recycling_strategy = Column(String)  # Mechanical Recycling, Chemical Recycling, Fabric Reuse, etc.
    
    # Environmental Impacts
    co2_savings = Column(Float)  # kg CO2
    water_savings = Column(Float)  # Liters
    landfill_reduction = Column(Float)  # kg
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    batch = relationship("WasteBatch", back_populates="analysis")
class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable means broadcast
    type = Column(String, nullable=False)  # collection, opportunity, milestone, warning, announcement
    message = Column(String, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="notifications")
def init_db():
    Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
