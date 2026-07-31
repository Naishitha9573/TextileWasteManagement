import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add Backend folder to path for import convenience
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from database import init_db
import algorithms

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Clean up any leftover test database state
    db_file = "./textile_waste.db"
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass
            
    init_db()
    
    # Explicitly seed test users
    from database import SessionLocal, User
    from auth import get_password_hash
    db = SessionLocal()
    try:
        default_users = [
            ("admin", "admin@textilewaste.org", "admin123", "Administrator"),
            ("recycler", "operator@recyclingfacility.com", "recycler123", "Recycling Facility Operator"),
            ("sustainability", "manager@sustainability.org", "sustainability123", "Sustainability Manager"),
            ("manufacturer", "waste@textilemanufacturer.com", "manufacturer123", "Textile Manufacturer"),
        ]
        for username, email, password, role in default_users:
            exists = db.query(User).filter(User.username == username).first()
            if not exists:
                hashed = get_password_hash(password)
                user = User(username=username, email=email, hashed_password=hashed, role=role)
                db.add(user)
        db.commit()
    finally:
        db.close()
    yield



# 1. TEST SCORING ALGORITHMS
def test_circularity_scoring():
    # Test high quality organic cotton
    scores = algorithms.calculate_scores(
        fabric_type="Cotton",
        condition="Excellent",
        category="Reusable",
        damage=False,
        contamination=False
    )
    
    assert scores["overall_circularity_score"] >= 85
    assert scores["circularity_category"] == "Excellent Recovery Potential"
    
    # Test contaminated polyester
    scores_bad = algorithms.calculate_scores(
        fabric_type="Polyester",
        condition="Contaminated",
        category="Hazardous Textile Waste",
        damage=True,
        contamination=True
    )
    
    assert scores_bad["overall_circularity_score"] < 30
    assert scores_bad["circularity_category"] == "Disposal Recommended"

def test_environmental_savings():
    # Test cotton environmental calculator
    savings = algorithms.calculate_environmental_impact("Cotton", 100.0, "Recyclable")
    
    assert savings["co2_savings"] == 220.0  # 100 * 2.2
    assert savings["water_savings"] == 250000.0  # 100 * 2500.0
    assert savings["landfill_reduction"] == 100.0
    
    # Test hazardous waste savings
    savings_haz = algorithms.calculate_environmental_impact("Cotton", 100.0, "Hazardous Textile Waste")
    assert savings_haz["co2_savings"] == 0.0
    assert savings_haz["water_savings"] == 0.0
    assert savings_haz["landfill_reduction"] == 0.0

# 2. TEST ENDPOINTS VIA FASTAPI TESTCLIENT
def test_auth_and_routing():
    # Test oauth registration/login mock endpoint
    oauth_payload = {
        "email": "test_pytest_user@textilewaste.org",
        "name": "Pytest OAuth User"
    }
    
    res = client.post("/api/auth/oauth-mock", json=oauth_payload)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "Recycling Facility Operator"
    
    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test fetch profile details
    res_profile = client.get("/api/auth/me", headers=headers)
    assert res_profile.status_code == 200
    assert res_profile.json()["username"] == "test_pytest_user"
    
    # Test list batches (should be empty/have seed batches depending on startup)
    res_batches = client.get("/api/batches", headers=headers)
    assert res_batches.status_code == 200
    batches = res_batches.json()
    assert len(batches) >= 0

def test_batch_creation_and_analysis():
    # Login as recycler
    login_res = client.post("/api/auth/token", json={"username": "recycler", "password": "recycler123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create batch
    batch_payload = {
        "fabric_type": "Silk",
        "source": "HighFashion Atelier",
        "quantity": 50.0,
        "color": "Emerald Green",
        "condition": "Excellent",
        "collection_date": "2026-07-24"
    }
    
    res_create = client.post("/api/batches", json=batch_payload, headers=headers)
    assert res_create.status_code == 200
    batch_data = res_create.json()
    batch_id = batch_data["id"]
    assert batch_data["status"] == "Registered"
    
    # Trigger image analysis (without image file, falling back to simulated extraction)
    res_analyze = client.post(f"/api/batches/{batch_id}/analyze", headers=headers)
    assert res_analyze.status_code == 200
    analyzed_data = res_analyze.json()
    
    assert analyzed_data["status"] == "Analyzed"
    assert analyzed_data["analysis"] is not None
    assert "overall_circularity_score" in analyzed_data["analysis"]
    assert analyzed_data["analysis"]["co2_savings"] > 0
    assert analyzed_data["analysis"]["water_savings"] > 0

def test_rbac_restriction():
    # Login as manufacturer
    login_res = client.post("/api/auth/token", json={"username": "manufacturer", "password": "manufacturer123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Try to access admin users endpoint (should fail)
    res_users = client.get("/api/users", headers=headers)
    assert res_users.status_code == 403
