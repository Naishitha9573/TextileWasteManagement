from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date, datetime

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
    classification_report: Optional[dict] = None
    recycling_options: Optional[str] = None

    class Config:
        from_attributes = True

class WasteBatchBase(BaseModel):
    fabric_type: str
    source: str
    quantity: float
    color: str
    condition: str
    collection_date: date

class WasteBatchCreate(WasteBatchBase):
    pass

class WasteBatchUpdate(BaseModel):
    fabric_type: Optional[str] = None
    source: Optional[str] = None
    quantity: Optional[float] = None
    color: Optional[str] = None
    condition: Optional[str] = None
    collection_date: Optional[date] = None
    status: Optional[str] = None

class WasteBatchResponse(WasteBatchBase):
    id: int
    status: str
    created_at: datetime
    user_id: Optional[int] = None
    analysis: Optional[AnalysisResultResponse] = None

    class Config:
        from_attributes = True


# Inventory Schemas (wraps WasteBatch with additional inventory fields)
class InventoryBase(BaseModel):
    batch_identifier: Optional[str] = None
    fabric_type: str
    source: str
    manufacturer: Optional[str] = None
    quantity: float
    unit: Optional[str] = "kg"
    color: Optional[str] = None
    condition: Optional[str] = None
    collection_date: str
    location: Optional[str] = None
    status: Optional[str] = "Registered"
    waste_category: Optional[str] = None
    notes: Optional[str] = None

class InventoryCreate(BaseModel):
    batch_identifier: Optional[str] = None
    fabric_type: str
    source: str
    manufacturer: Optional[str] = None
    quantity: float
    unit: Optional[str] = "kg"
    color: Optional[str] = None
    condition: Optional[str] = None
    collection_date: date
    location: Optional[str] = None
    status: Optional[str] = "Registered"
    waste_category: Optional[str] = None
    notes: Optional[str] = None


class InventoryUpdate(BaseModel):
    batch_identifier: Optional[str] = None
    fabric_type: Optional[str] = None
    source: Optional[str] = None
    manufacturer: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    color: Optional[str] = None
    condition: Optional[str] = None
    collection_date: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    waste_category: Optional[str] = None
    notes: Optional[str] = None

class InventoryResponse(BaseModel):
    id: int
    batch_identifier: Optional[str] = None

    fabric_type: str
    source: str
    manufacturer: Optional[str] = None

    quantity: float
    unit: str

    color: Optional[str] = None
    condition: Optional[str] = None

    collection_date: date

    location: Optional[str] = None

    status: str
    waste_category: Optional[str] = None
    notes: Optional[str] = None

    user_id: int

    created_at: datetime

    updated_at: Optional[datetime] = None


    class Config:
        from_attributes = True 

class InventoryStatistics(BaseModel):
    total_inventory: int
    total_quantity: float
    pending: int
    processed: int
    recyclable: int
    reusable: int
    cotton: int
    polyester: int
    denim: int
    mixed_fabric: int

class MaterialPredictionRequest(BaseModel):
    fabric_type: str
    condition: str
    quantity: float
    source: Optional[str] = None
    color: Optional[str] = None
    damage: bool = False
    contamination: bool = False

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
