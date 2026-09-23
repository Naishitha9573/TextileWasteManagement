from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime
import database
import schemas
import auth
from database import get_db, WasteBatch, Notification
from auth import get_current_user

router = APIRouter()

@router.post("/api/inventory", response_model=schemas.InventoryResponse)
def create_inventory(item_in: schemas.InventoryCreate, current_user: database.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Role check
    if current_user.role not in ["Recycling Facility Operator", "Textile Manufacturer", "Administrator"]:
        raise HTTPException(status_code=403, detail="Not authorized to create inventory")

    # Validation
    if item_in.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than zero")
    if not item_in.collection_date:
        raise HTTPException(status_code=400, detail="Collection date required")
    if not item_in.fabric_type:
        raise HTTPException(status_code=400, detail="Fabric type required")
    if not item_in.source:
        raise HTTPException(status_code=400, detail="Source required")

    # Unique batch identifier check
    if item_in.batch_identifier:
        existing = db.query(WasteBatch).filter(WasteBatch.batch_identifier == item_in.batch_identifier).first()
        if existing:
            raise HTTPException(status_code=400, detail="Batch identifier already exists")

    batch = WasteBatch(
        batch_identifier=item_in.batch_identifier,
        fabric_type=item_in.fabric_type,
        source=item_in.source,
        manufacturer=item_in.manufacturer,
        quantity=item_in.quantity,
        unit=item_in.unit or "kg",
        color=item_in.color or "",
        condition=item_in.condition or "",
        collection_date=item_in.collection_date,
        location=item_in.location,
        status=item_in.status or "Registered",
        waste_category=item_in.waste_category or "",
        notes=item_in.notes or "",
        user_id=current_user.id
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    # Notification
    notif = Notification(type="collection", message=f"Inventory Created: Batch #{batch.id} - {batch.fabric_type}")
    db.add(notif)
    db.commit()

    return batch

@router.get("/api/inventory", response_model=List[schemas.InventoryResponse])
def list_inventory(
    page: int = 1,
    page_size: int = 20,
    sort_by: Optional[str] = "id",
    sort_order: Optional[str] = "desc",
    current_user: database.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    fabric_type: Optional[str] = None,
    status: Optional[str] = None,
    condition: Optional[str] = None,
):
    query = db.query(WasteBatch)
    # Role-based filtering
    if current_user.role == "Textile Manufacturer":
        query = query.filter(WasteBatch.user_id == current_user.id)

    if fabric_type:
        query = query.filter(WasteBatch.fabric_type == fabric_type)
    if status:
        query = query.filter(WasteBatch.status == status)
    if condition:
        query = query.filter(WasteBatch.condition == condition)

    # Sorting
    if sort_order.lower() == "desc":
        query = query.order_by(getattr(WasteBatch, sort_by).desc())
    else:
        query = query.order_by(getattr(WasteBatch, sort_by))

    offset = (page - 1) * page_size
    items = query.offset(offset).limit(page_size).all()
    return items

@router.get("/api/inventory/{id}", response_model=schemas.InventoryResponse)
def get_inventory(id: int, current_user: database.User = Depends(get_current_user), db: Session = Depends(get_db)):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Inventory not found")
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return batch

@router.put("/api/inventory/{id}", response_model=schemas.InventoryResponse)
def update_inventory(id: int, item_in: schemas.InventoryUpdate, current_user: database.User = Depends(get_current_user), db: Session = Depends(get_db)):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Inventory not found")
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    data = item_in.dict(exclude_unset=True)
    if "quantity" in data and data["quantity"] is not None and data["quantity"] <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than zero")
    # Batch identifier uniqueness
    if "batch_identifier" in data and data.get("batch_identifier"):
        existing = db.query(WasteBatch).filter(WasteBatch.batch_identifier == data.get("batch_identifier"), WasteBatch.id != id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Batch identifier already exists")

    for field, value in data.items():
        setattr(batch, field, value)
    batch.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(batch)
    return batch

@router.delete("/api/inventory/{id}")
def delete_inventory(id: int, current_user: database.User = Depends(get_current_user), db: Session = Depends(get_db)):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Inventory not found")
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    db.delete(batch)
    db.commit()
    return {"message": f"Inventory {id} deleted successfully"}

@router.get("/api/inventory/search", response_model=List[schemas.InventoryResponse])
def search_inventory(query: str, page: int = 1, page_size: int = 20, db: Session = Depends(get_db), current_user: database.User = Depends(get_current_user)):
    q = db.query(WasteBatch).filter(
        (WasteBatch.batch_identifier.ilike(f"%{query}%")) |
        (WasteBatch.fabric_type.ilike(f"%{query}%")) |
        (WasteBatch.manufacturer.ilike(f"%{query}%")) |
        (WasteBatch.source.ilike(f"%{query}%")) |
        (WasteBatch.location.ilike(f"%{query}%")) |
        (WasteBatch.waste_category.ilike(f"%{query}%")) |
        (WasteBatch.notes.ilike(f"%{query}%")) |
        (WasteBatch.color.ilike(f"%{query}%"))
    )
    if current_user.role == "Textile Manufacturer":
        q = q.filter(WasteBatch.user_id == current_user.id)
    offset = (page - 1) * page_size
    return q.offset(offset).limit(page_size).all()

@router.get("/api/inventory/filter", response_model=List[schemas.InventoryResponse])
def filter_inventory(
    fabric_type: Optional[str] = None,
    status: Optional[str] = None,
    condition: Optional[str] = None,
    source: Optional[str] = None,
    manufacturer: Optional[str] = None,
    waste_category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    current_user: database.User = Depends(get_current_user)
):
    q = db.query(WasteBatch)
    if current_user.role == "Textile Manufacturer":
        q = q.filter(WasteBatch.user_id == current_user.id)
    if fabric_type:
        q = q.filter(WasteBatch.fabric_type == fabric_type)
    if status:
        q = q.filter(WasteBatch.status == status)
    if condition:
        q = q.filter(WasteBatch.condition == condition)
    if source:
        q = q.filter(WasteBatch.source == source)
    if manufacturer:
        q = q.filter(WasteBatch.manufacturer == manufacturer)
    if waste_category:
        q = q.filter(WasteBatch.waste_category == waste_category)
    if date_from:
        q = q.filter(WasteBatch.collection_date >= date_from)
    if date_to:
        q = q.filter(WasteBatch.collection_date <= date_to)

    offset = (page - 1) * page_size
    return q.offset(offset).limit(page_size).all()

@router.get("/api/inventory/statistics", response_model=schemas.InventoryStatistics)
def inventory_statistics(db: Session = Depends(get_db), current_user: database.User = Depends(get_current_user)):
    # Role-based access: everyone can view
    total_inventory = db.query(WasteBatch).count()
    total_quantity = db.query(func.sum(WasteBatch.quantity)).scalar() or 0.0
    pending = db.query(WasteBatch).filter(WasteBatch.status == "Registered").count()
    processed = db.query(WasteBatch).filter(WasteBatch.status == "Processed").count()
    recyclable = db.query(WasteBatch).filter(WasteBatch.fabric_type.ilike("%cotton%") | WasteBatch.fabric_type.ilike("%polyester%") | WasteBatch.fabric_type.ilike("%denim%") | WasteBatch.fabric_type.ilike("%mixed%") ).count()
    reusable = db.query(WasteBatch).filter(WasteBatch.condition.ilike("%excellent%") | WasteBatch.condition.ilike("%good%") ).count()

    cotton = db.query(WasteBatch).filter(WasteBatch.fabric_type.ilike("%cotton%")).count()
    polyester = db.query(WasteBatch).filter(WasteBatch.fabric_type.ilike("%polyester%")).count()
    denim = db.query(WasteBatch).filter(WasteBatch.fabric_type.ilike("%denim%")).count()
    mixed_fabric = db.query(WasteBatch).filter(WasteBatch.fabric_type.ilike("%mixed%") | WasteBatch.fabric_type.ilike("%mix%") ).count()

    return {
        "total_inventory": total_inventory,
        "total_quantity": float(total_quantity),
        "pending": pending,
        "processed": processed,
        "recyclable": recyclable,
        "reusable": reusable,
        "cotton": cotton,
        "polyester": polyester,
        "denim": denim,
        "mixed_fabric": mixed_fabric
    }
