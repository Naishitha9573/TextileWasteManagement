from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserRoleUpdate(BaseModel):
    role: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Analysis Result Schemas
class AnalysisResultResponse(BaseModel):
    id: int
    batch_id: int
    fabric_texture: str
    fabric_pattern: str
    fabric_color: str
    damage_detected: bool
    contamination_detected: bool
    recyclability_score: float
    reuse_score: float
    sustainability_score: float
    material_recovery_score: float
    overall_circularity_score: float
    circularity_category: str
    recycling_strategy: str
    co2_savings: float
    water_savings: float
    landfill_reduction: float
    created_at: datetime

    class Config:
        from_attributes = True

# Waste Batch Schemas
class WasteBatchBase(BaseModel):
    fabric_type: str
    source: str
    quantity: float
    color: str
    condition: str
    collection_date: str

class WasteBatchCreate(WasteBatchBase):
    pass

class WasteBatchUpdate(BaseModel):
    fabric_type: Optional[str] = None
    source: Optional[str] = None
    quantity: Optional[float] = None
    color: Optional[str] = None
    condition: Optional[str] = None
    collection_date: Optional[str] = None
    status: Optional[str] = None

class WasteBatchResponse(WasteBatchBase):
    id: int
    status: str
    created_at: datetime
    user_id: int
    analysis: Optional[AnalysisResultResponse] = None

    class Config:
        from_attributes = True

# Notification Schemas
class NotificationBase(BaseModel):
    type: str
    message: str

class NotificationCreate(NotificationBase):
    user_id: Optional[int] = None

class NotificationResponse(NotificationBase):
    id: int
    user_id: Optional[int]
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Analytics Response Schemas
class RecyclerAnalytics(BaseModel):
    total_batches: int
    total_quantity_kg: float
    recycled_count: int
    processed_count: int
    category_distribution: dict
    fabric_distribution: dict
    recovery_stats: dict

class SustainabilityAnalytics(BaseModel):
    co2_saved_kg: float
    water_saved_liters: float
    landfill_diverted_kg: float
    diversion_rate: float
    circularity_avg: float
    milestones: List[dict]

class ManufacturerAnalytics(BaseModel):
    waste_generated_kg: float
    average_circularity: float
    recycled_percentage: float
    co2_savings_kg: float
    waste_by_source: dict

class AdminAnalytics(BaseModel):
    total_users: int
    total_batches: int
    active_connections: int
    system_status: str
    database_size_bytes: int
