import os
import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text
# Local imports
import database
import schemas
import auth
import algorithms
import reports
import mongo
from database import get_db, init_db, User, WasteBatch, AnalysisResult, Notification
from auth import get_current_user, RoleChecker, get_password_hash, verify_password, create_access_token
app = FastAPI(title="Textile Waste Intelligence Platform API")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup_event():
    # Initialize Database tables
    init_db()
    
    # Initialize MongoDB connection as secondary database
    try:
        mongo.init_mongo()
    except Exception as e:
        print(f"MongoDB startup connection failed: {e}")

    # Seed default users if they do not exist
    db = next(get_db())
    try:
        users_count = db.query(User).count()
        if users_count == 0:
            # Create default roles
            default_users = [
                ("admin", "admin@textilewaste.org", "admin123", "Administrator"),
                ("recycler", "operator@recyclingfacility.com", "recycler123", "Recycling Facility Operator"),
                ("sustainability", "manager@sustainability.org", "sustainability123", "Sustainability Manager"),
                ("manufacturer", "waste@textilemanufacturer.com", "manufacturer123", "Textile Manufacturer"),
            ]
            for username, email, password, role in default_users:
                hashed = get_password_hash(password)
                user = User(username=username, email=email, hashed_password=hashed, role=role)
                db.add(user)
            db.commit()
            
            # Seed sample waste batches
            seed_batches(db)
            
            # Seed notifications
            seed_notifications(db)
            
            print("Database successfully seeded.")
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        db.close()
def seed_batches(db: Session):
    # Fetch recycler user to associate batches
    recycler = db.query(User).filter(User.username == "recycler").first()
    manufacturer = db.query(User).filter(User.username == "manufacturer").first()
    
    u_id = recycler.id if recycler else 1
    m_id = manufacturer.id if manufacturer else 2
    
    sample_data = [
        # fabric_type, source, quantity, color, condition, collection_date, user_id, status
        ("Cotton", "EcoThread Garments", 180.0, "Crimson Red", "Excellent", "2026-07-20", m_id, "Analyzed"),
        ("Polyester", "PolyWeave Textiles", 340.0, "Navy Blue", "Poor", "2026-07-21", u_id, "Analyzed"),
        ("Wool", "Highland Shearers", 75.0, "Cream White", "Good", "2026-07-22", u_id, "Analyzed"),
        ("Denim", "Indigo Denim Inc", 250.0, "Dark Denim", "Fair", "2026-07-23", m_id, "Analyzed"),
        ("Mixed Fabrics", "Consumer Drop-off Box", 110.0, "Multi-color", "Contaminated", "2026-07-23", u_id, "Analyzed"),
        ("Linen", "FlaxFields Clothing", 90.0, "Natural Beige", "Good", "2026-07-24", m_id, "Registered")
    ]
    
    for fabric_type, source, qty, color, cond, col_date, user, status in sample_data:
        batch = WasteBatch(
            fabric_type=fabric_type,
            source=source,
            quantity=qty,
            color=color,
            condition=cond,
            collection_date=col_date,
            status=status,
            user_id=user
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        
        # If status is Analyzed, compute and insert analysis result
        if status == "Analyzed":
            # Extract features simulated
            features = {
                "fabric_texture": "Smooth / Soft" if fabric_type in ["Cotton", "Silk", "Linen"] else ("Coarse / Woven" if fabric_type == "Denim" else "Fine Fiber"),
                "fabric_pattern": "Solid Color" if fabric_type != "Mixed Fabrics" else "Melange",
                "fabric_color": color,
                "damage_detected": cond in ["Poor", "Contaminated"],
                "contamination_detected": cond == "Contaminated"
            }
            
            comp_q = algorithms.get_composition_and_quality(fabric_type, cond)
            w_cat = algorithms.get_waste_classification(cond, features["damage_detected"], features["contamination_detected"])
            recs = algorithms.get_recycling_recommendations(fabric_type, w_cat)
            scores = algorithms.calculate_scores(fabric_type, cond, w_cat, features["damage_detected"], features["contamination_detected"])
            env_impacts = algorithms.calculate_environmental_impact(fabric_type, qty, w_cat)
            
            analysis = AnalysisResult(
                batch_id=batch.id,
                fabric_texture=features["fabric_texture"],
                fabric_pattern=features["fabric_pattern"],
                fabric_color=features["fabric_color"],
                damage_detected=features["damage_detected"],
                contamination_detected=features["contamination_detected"],
                recyclability_score=scores["recyclability_score"],
                reuse_score=scores["reuse_score"],
                sustainability_score=scores["sustainability_score"],
                material_recovery_score=scores["material_recovery_score"],
                overall_circularity_score=scores["overall_circularity_score"],
                circularity_category=scores["circularity_category"],
                recycling_strategy=f"{recs['strategy']}: {recs['options']}",
                co2_savings=env_impacts["co2_savings"],
                water_savings=env_impacts["water_savings"],
                landfill_reduction=env_impacts["landfill_reduction"]
            )
            db.add(analysis)
            db.commit()
def seed_notifications(db: Session):
    notifications = [
        ("warning", "High-contamination warning: Batch #5 contains chemical pollutants, sorting required."),
        ("milestone", "Sustainability Milestone: Diversion rate has exceeded 80% this week!"),
        ("collection", "New collection schedule: 120 kg Linen registered by FlaxFields Clothing."),
        ("opportunity", "Upcycling opportunity: Cotton batch #1 matches requirements for local repair boutique.")
    ]
    for n_type, msg in notifications:
        notif = Notification(type=n_type, message=msg)
        db.add(notif)
    db.commit()
# --- AUTHENTICATION ENDPOINTS ---
@app.post("/api/auth/register", response_model=schemas.UserResponse)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user_in.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = db.query(User).filter(User.email == user_in.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed = get_password_hash(user_in.password)
    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed,
        role=user_in.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
@app.post("/api/auth/token", response_model=schemas.Token)
def login(form_data: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "username": user.username
    }
@app.post("/api/auth/oauth-mock", response_model=schemas.Token)
def oauth_mock(provider_data: dict, db: Session = Depends(get_db)):
    # Simulates OAuth2 (Google/GitHub) callback
    email = provider_data.get("email")
    name = provider_data.get("name", "oauth_user")
    if not email:
        raise HTTPException(status_code=400, detail="OAuth email not provided")
        
    # Check if user exists, otherwise create
    user = db.query(User).filter(User.email == email).first()
    if not user:
        username = email.split("@")[0]
        # Append some random characters to avoid collision
        counter = 1
        original_username = username
        while db.query(User).filter(User.username == username).first():
            username = f"{original_username}{counter}"
            counter += 1
            
        hashed = get_password_hash("OAuthMockPasswordSecure123")
        user = User(
            username=username,
            email=email,
            hashed_password=hashed,
            role="Recycling Facility Operator"  # Default role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "username": user.username
    }
@app.get("/api/auth/me", response_model=schemas.UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
# --- WASTE INVENTORY ENDPOINTS ---
@app.get("/api/batches", response_model=List[schemas.WasteBatchResponse])
def list_batches(
    status: Optional[str] = None, 
    fabric_type: Optional[str] = None, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    query = db.query(WasteBatch)
    
    # Filter based on role: Textile Manufacturers only view their own registered batches
    if current_user.role == "Textile Manufacturer":
        query = query.filter(WasteBatch.user_id == current_user.id)
        
    if status:
        query = query.filter(WasteBatch.status == status)
    if fabric_type:
        query = query.filter(WasteBatch.fabric_type == fabric_type)
        
    return query.order_by(WasteBatch.id.desc()).all()
@app.post("/api/batches", response_model=schemas.WasteBatchResponse)
def create_batch(
    batch_in: schemas.WasteBatchCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Only Recyclers, Manufacturers, and Admins can create batches
    if current_user.role not in ["Recycling Facility Operator", "Textile Manufacturer", "Administrator"]:
        raise HTTPException(status_code=403, detail="Not authorized to log inventory batches")
        
    batch = WasteBatch(
        fabric_type=batch_in.fabric_type,
        source=batch_in.source,
        quantity=batch_in.quantity,
        color=batch_in.color,
        condition=batch_in.condition,
        collection_date=batch_in.collection_date,
        status="Registered",
        user_id=current_user.id
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    
    # Add a notification
    notif = Notification(
        type="collection",
        message=f"New Batch #{batch.id} registered: {batch.quantity}kg of {batch.fabric_type} from {batch.source}."
    )
    db.add(notif)
    db.commit()
    
    return batch
@app.get("/api/batches/{id}", response_model=schemas.WasteBatchResponse)
def get_batch(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    # Check permissions
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to other manufacturer batches")
        
    return batch
@app.put("/api/batches/{id}", response_model=schemas.WasteBatchResponse)
def update_batch(
    id: int, 
    batch_in: schemas.WasteBatchUpdate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    for field, value in batch_in.dict(exclude_unset=True).items():
        setattr(batch, field, value)
        
    db.commit()
    db.refresh(batch)
    return batch
@app.delete("/api/batches/{id}")
def delete_batch(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    if current_user.role == "Textile Manufacturer" and batch.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    db.delete(batch)
    db.commit()
    return {"message": f"Batch {id} deleted successfully"}
# --- IMAGE ANALYSIS & MACHINE LEARNING CLASSIFICATION ---
@app.post("/api/batches/{id}/analyze", response_model=schemas.WasteBatchResponse)
async def analyze_batch_image(
    id: int,
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve the batch
    batch = db.query(WasteBatch).filter(WasteBatch.id == id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
        
    # Read files for analysis
    image_bytes = None
    filename = "simulated_upload.png"
    if file:
        image_bytes = await file.read()
        filename = file.filename
        
    # 1. Computer Vision Image Analysis
    cv_features = algorithms.analyze_image(image_bytes, filename)
    
    # 2. Material Classification (Purity, Blends, Quality)
    comp_q = algorithms.get_composition_and_quality(batch.fabric_type, batch.condition)
    
    # Override texture details if cv extracts them
    # Determine waste category
    w_category = algorithms.get_waste_classification(
        batch.condition, 
        cv_features["damage_detected"], 
        cv_features["contamination_detected"]
    )
    
    # 3. Recommendations & Environmental Calculations
    recs = algorithms.get_recycling_recommendations(batch.fabric_type, w_category)
    scores = algorithms.calculate_scores(
        batch.fabric_type, 
        batch.condition, 
        w_category, 
        cv_features["damage_detected"], 
        cv_features["contamination_detected"]
    )
    env_impacts = algorithms.calculate_environmental_impact(batch.fabric_type, batch.quantity, w_category)
    
    # Create or update AnalysisResult
    analysis = db.query(AnalysisResult).filter(AnalysisResult.batch_id == batch.id).first()
    if not analysis:
        analysis = AnalysisResult(batch_id=batch.id)
        
    analysis.fabric_texture = cv_features["fabric_texture"]
    analysis.fabric_pattern = cv_features["fabric_pattern"]
    analysis.fabric_color = cv_features["fabric_color"]
    analysis.damage_detected = cv_features["damage_detected"]
    analysis.contamination_detected = cv_features["contamination_detected"]
    
    # Save scores
    analysis.recyclability_score = scores["recyclability_score"]
    analysis.reuse_score = scores["reuse_score"]
    analysis.sustainability_score = scores["sustainability_score"]
    analysis.material_recovery_score = scores["material_recovery_score"]
    analysis.overall_circularity_score = scores["overall_circularity_score"]
    analysis.circularity_category = scores["circularity_category"]
    
    # Save recommendation & environmental impact
    analysis.recycling_strategy = f"{recs['strategy']}: {recs['options']}"
    analysis.co2_savings = env_impacts["co2_savings"]
    analysis.water_savings = env_impacts["water_savings"]
    analysis.landfill_reduction = env_impacts["landfill_reduction"]
    
    # Update batch status
    batch.status = "Analyzed"
    
    db.add(analysis)
    db.commit()
    
    # Add relevant notifications
    if cv_features["contamination_detected"]:
        notif = Notification(
            type="warning",
            message=f"CRITICAL Warning: Contamination detected in Batch #{batch.id} ({batch.fabric_type}). Special disposal recommended."
        )
        db.add(notif)
        
    if scores["overall_circularity_score"] >= 85:
        notif = Notification(
            type="opportunity",
            message=f"Opportunity Alert: Batch #{batch.id} has EXCELLENT recovery potential ({scores['overall_circularity_score']}%). Directing to Premium Recyclers."
        )
        db.add(notif)
        
    db.commit()
    db.refresh(batch)
    return batch
# --- NOTIFICATION ENDPOINTS ---
@app.get("/api/notifications", response_model=List[schemas.NotificationResponse])
def get_notifications(
    unread_only: bool = False, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    query = db.query(Notification).filter(
        (Notification.user_id == current_user.id) | (Notification.user_id == None)
    )
    if unread_only:
        query = query.filter(Notification.is_read == False)
    return query.order_by(Notification.id.desc()).all()
@app.put("/api/notifications/{id}/read")
def read_notification(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    notif = db.query(Notification).filter(Notification.id == id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
        
    notif.is_read = True
    db.commit()
    return {"status": "success"}
# --- ANALYTICS & DASHBOARD ENDPOINTS ---
@app.get("/api/analytics/recycler", response_model=schemas.RecyclerAnalytics)
def get_recycler_analytics(
    current_user: User = Depends(auth.get_current_user), 
    db: Session = Depends(get_db)
):
    # Requires Recycler or Admin
    if current_user.role not in ["Recycling Facility Operator", "Administrator", "Sustainability Manager"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    total_batches = db.query(WasteBatch).count()
    total_qty = db.query(func.sum(WasteBatch.quantity)).scalar() or 0.0
    recycled_count = db.query(WasteBatch).filter(WasteBatch.status == "Processed").count()
    processed_count = db.query(WasteBatch).filter(WasteBatch.status == "Analyzed").count()
    
    # Calculate category distributions
    categories = db.query(AnalysisResult.circularity_category, func.count(AnalysisResult.id))\
                   .group_by(AnalysisResult.circularity_category).all()
    cat_dist = {cat or "Unanalyzed": count for cat, count in categories}
    
    # Fabric distributions
    fabrics = db.query(WasteBatch.fabric_type, func.count(WasteBatch.id)).group_by(WasteBatch.fabric_type).all()
    fab_dist = {fab: count for fab, count in fabrics}
    
    # Recovery Stats
    recycled_weight = db.query(func.sum(WasteBatch.quantity)).filter(WasteBatch.status.in_(["Analyzed", "Processed"])).scalar() or 0.0
    
    recovery_stats = {
        "registered_qty": total_qty,
        "processed_qty": recycled_weight,
        "diverted_percentage": (recycled_weight / total_qty * 100) if total_qty > 0 else 0.0
    }
    
    return {
        "total_batches": total_batches,
        "total_quantity_kg": total_qty,
        "recycled_count": recycled_count,
        "processed_count": processed_count,
        "category_distribution": cat_dist,
        "fabric_distribution": fab_dist,
        "recovery_stats": recovery_stats
    }
@app.get("/api/analytics/sustainability", response_model=schemas.SustainabilityAnalytics)
def get_sustainability_analytics(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # Environmental metrics aggregate
    co2 = db.query(func.sum(AnalysisResult.co2_savings)).scalar() or 0.0
    water = db.query(func.sum(AnalysisResult.water_savings)).scalar() or 0.0
    landfill = db.query(func.sum(AnalysisResult.landfill_reduction)).scalar() or 0.0
    
    total_qty = db.query(func.sum(WasteBatch.quantity)).scalar() or 1.0
    diverted_qty = db.query(func.sum(WasteBatch.quantity)).filter(WasteBatch.status.in_(["Analyzed", "Processed"])).scalar() or 0.0
    
    avg_circ = db.query(func.avg(AnalysisResult.overall_circularity_score)).scalar() or 0.0
    
    # Generate some milestones
    milestones = [
        {"title": "Carbon Neutral Step", "desc": "Saved 500+ kg of CO2 equivalent emissions.", "achieved": co2 >= 500},
        {"title": "Water Conservator", "desc": "Conserved 100,000+ liters of water.", "achieved": water >= 100000},
        {"title": "Landfill Savior", "desc": "Diverted over 1 ton of textile waste.", "achieved": landfill >= 1000},
    ]
    
    return {
        "co2_saved_kg": round(co2, 1),
        "water_saved_liters": round(water, 1),
        "landfill_diverted_kg": round(landfill, 1),
        "diversion_rate": round((diverted_qty / total_qty * 100), 1) if total_qty > 0 else 0.0,
        "circularity_avg": round(avg_circ, 1),
        "milestones": milestones
    }
@app.get("/api/analytics/manufacturer", response_model=schemas.ManufacturerAnalytics)
def get_manufacturer_analytics(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    # Requires Manufacturer or Admin
    if current_user.role not in ["Textile Manufacturer", "Administrator"]:
        raise HTTPException(status_code=403, detail="Access denied")
        
    query = db.query(WasteBatch).filter(WasteBatch.user_id == current_user.id)
    
    total_waste = query.with_entities(func.sum(WasteBatch.quantity)).scalar() or 0.0
    
    # Average circularity of analyzed manufacturer batches
    avg_circ = db.query(func.avg(AnalysisResult.overall_circularity_score))\
                 .join(WasteBatch)\
                 .filter(WasteBatch.user_id == current_user.id).scalar() or 0.0
                 
    # Recycled count
    analyzed_count = query.filter(WasteBatch.status.in_(["Analyzed", "Processed"])).count()
    total_count = query.count()
    recycled_pct = (analyzed_count / total_count * 100) if total_count > 0 else 0.0
    
    # Environmental savings
    co2 = db.query(func.sum(AnalysisResult.co2_savings)).join(WasteBatch).filter(WasteBatch.user_id == current_user.id).scalar() or 0.0
            
    # Waste breakdown by fabric type
    fabrics = query.with_entities(WasteBatch.fabric_type, func.count(WasteBatch.id)).group_by(WasteBatch.fabric_type).all()
    by_source = {fab: count for fab, count in fabrics}
    
    return {
        "waste_generated_kg": total_waste,
        "average_circularity": round(avg_circ, 1),
        "recycled_percentage": round(recycled_pct, 1),
        "co2_savings_kg": round(co2, 1),
        "waste_by_source": by_source
    }
@app.get("/api/analytics/admin", response_model=schemas.AdminAnalytics)
def get_admin_analytics(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if current_user.role != "Administrator":
        raise HTTPException(status_code=403, detail="Access restricted to Administrators")
        
    total_users = db.query(User).count()
    total_batches = db.query(WasteBatch).count()
    
    return {
        "total_users": total_users,
        "total_batches": total_batches,
        "active_connections": 12,  # Simulated active users
        "system_status": "Healthy / Operational",
        "database_size_bytes": os.path.getsize("./textile_waste.db") if os.path.exists("./textile_waste.db") else 1024
    }
# --- USER MANAGEMENT ENDPOINTS (ADMIN ONLY) ---
@app.get("/api/users", response_model=List[schemas.UserResponse])
def list_users(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if current_user.role != "Administrator":
        raise HTTPException(status_code=403, detail="Access denied")
    return db.query(User).order_by(User.id.desc()).all()
@app.put("/api/users/{id}/role", response_model=schemas.UserResponse)
def update_user_role(
    id: int, 
    role_update: schemas.UserRoleUpdate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if current_user.role != "Administrator":
        raise HTTPException(status_code=403, detail="Access denied")
        
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.role = role_update.role
    db.commit()
    db.refresh(user)
    return user
@app.delete("/api/users/{id}")
def delete_user(id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role != "Administrator":
        raise HTTPException(status_code=403, detail="Access denied")
        
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Prevent deleting oneself
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
        
    db.delete(user)
    db.commit()
    return {"message": f"User {user.username} deleted successfully"}


# --- REPORTS ENDPOINTS (MODULE 13) ---
@app.get("/api/reports/pdf")
def export_pdf_report(
    status_filter: Optional[str] = None,
    fabric_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export waste batches as PDF report.
    
    Query Parameters:
    - status_filter: Filter by batch status (Registered, Analyzed, Processed)
    - fabric_type: Filter by fabric type
    """
    query = db.query(WasteBatch)
    
    # Apply role-based filtering
    if current_user.role == "Textile Manufacturer":
        query = query.filter(WasteBatch.user_id == current_user.id)
    
    # Apply filters
    if status_filter:
        query = query.filter(WasteBatch.status == status_filter)
    if fabric_type:
        query = query.filter(WasteBatch.fabric_type == fabric_type)
    
    batches = query.order_by(WasteBatch.id.desc()).all()
    
    if not batches:
        raise HTTPException(status_code=404, detail="No batches found matching criteria")
    
    # Generate PDF
    pdf_content = reports.ReportGenerator.generate_pdf_report(
        batches=batches,
        report_title="Textile Waste Intelligence Report",
        user_name=current_user.username
    )
    
    return StreamingResponse(
        iter([pdf_content]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=textile_waste_report.pdf"}
    )


@app.get("/api/reports/excel")
def export_excel_report(
    status_filter: Optional[str] = None,
    fabric_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export waste batches as Excel (XLSX) report.
    
    Query Parameters:
    - status_filter: Filter by batch status (Registered, Analyzed, Processed)
    - fabric_type: Filter by fabric type
    """
    query = db.query(WasteBatch)
    
    # Apply role-based filtering
    if current_user.role == "Textile Manufacturer":
        query = query.filter(WasteBatch.user_id == current_user.id)
    
    # Apply filters
    if status_filter:
        query = query.filter(WasteBatch.status == status_filter)
    if fabric_type:
        query = query.filter(WasteBatch.fabric_type == fabric_type)
    
    batches = query.order_by(WasteBatch.id.desc()).all()
    
    if not batches:
        raise HTTPException(status_code=404, detail="No batches found matching criteria")
    
    # Generate Excel
    excel_content = reports.ReportGenerator.generate_excel_report(
        batches=batches,
        report_title="Textile Waste Intelligence Report"
    )
    
    return StreamingResponse(
        iter([excel_content]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=textile_waste_report.xlsx"}
    )


@app.get("/api/reports/csv")
def export_csv_report(
    status_filter: Optional[str] = None,
    fabric_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export waste batches as CSV report.
    
    Query Parameters:
    - status_filter: Filter by batch status (Registered, Analyzed, Processed)
    - fabric_type: Filter by fabric type
    """
    query = db.query(WasteBatch)
    
    # Apply role-based filtering
    if current_user.role == "Textile Manufacturer":
        query = query.filter(WasteBatch.user_id == current_user.id)
    
    # Apply filters
    if status_filter:
        query = query.filter(WasteBatch.status == status_filter)
    if fabric_type:
        query = query.filter(WasteBatch.fabric_type == fabric_type)
    
    batches = query.order_by(WasteBatch.id.desc()).all()
    
    if not batches:
        raise HTTPException(status_code=404, detail="No batches found matching criteria")
    
    # Generate CSV
    csv_content = reports.ReportGenerator.generate_csv_report(batches=batches)
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=textile_waste_report.csv"}
    )


# Health check endpoint
@app.get("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    postgres_ok = False
    mongodb_ok = False

    try:
        with database.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            postgres_ok = True
    except Exception:
        postgres_ok = False

    try:
        mongo_db = mongo.get_mongo_db()
        mongo_db.command("ping")
        mongodb_ok = True
    except Exception:
        mongodb_ok = False

    return {
        "status": "healthy",
        "service": "Textile Waste Intelligence Platform API",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "postgresql_connected": postgres_ok,
        "mongodb_connected": mongodb_ok
    }

