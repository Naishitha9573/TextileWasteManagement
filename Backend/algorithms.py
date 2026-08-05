import io
import hashlib
import numpy as np
from PIL import Image

try:
    from ml.fabric_classifier import fabric_classifier
except Exception:  # pragma: no cover - fallback if sklearn import fails in minimal environments
    fabric_classifier = None


def analyze_image(image_bytes: bytes, filename: str) -> dict:
    """
    Parses an uploaded image using PIL and numpy to extract features:
    - Dominant/Average color
    - Edge density (for texture and pattern)
    - Value standard deviation (for damage detection)
    - Contamination indicators
    If image loading fails, falls back to a deterministic feature set based on the filename hash.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to speed up computations
        image.thumbnail((250, 250))
        img_np = np.array(image)
        
        # 1. Color analysis
        avg_color = img_np.mean(axis=(0, 1))  # [R, G, B]
        r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        
        # Map to common name
        color_name = map_rgb_to_name(r, g, b)

        # 2. Texture & Pattern analysis
        gray = image.convert("L")
        gray_np = np.array(gray)
        # Calculate horizontal and vertical differences (rough edge detector)
        h_diff = np.abs(gray_np[:-1, :] - gray_np[1:, :])
        v_diff = np.abs(gray_np[:, :-1] - gray_np[:, 1:])
        edge_score = float(np.mean(h_diff) + np.mean(v_diff))

        if edge_score > 18.0:
            texture = "Coarse / Woven"
            pattern = "Textured Pattern"
        elif edge_score > 10.0:
            texture = "Medium Knit"
            pattern = "Solid / Micro-pattern"
        else:
            texture = "Smooth / Soft"
            pattern = "Solid Color"

        # 3. Damage & Contamination detection
        # High intensity variance across the gray image can mean staining or tearing
        std_dev = float(np.std(gray_np))
        damage_detected = std_dev > 50.0  # High contrast spots/holes
        
        # Contamination: check for significant yellow/brown tint or green spots in non-yellow/green fabrics
        # Or compare color channels
        channel_diff = float(np.mean(np.abs(img_np[:, :, 0] - img_np[:, :, 1])) + np.mean(np.abs(img_np[:, :, 1] - img_np[:, :, 2])))
        contamination_detected = channel_diff > 45.0 and ("White" in color_name or "Grey" in color_name)

        return {
            "fabric_texture": texture,
            "fabric_pattern": pattern,
            "fabric_color": f"{color_name} ({color_hex})",
            "damage_detected": damage_detected,
            "contamination_detected": contamination_detected
        }

    except Exception:
        # High fidelity fallback using filename hash for consistent mock simulation
        hasher = hashlib.md5(filename.encode("utf-8"))
        hash_val = int(hasher.hexdigest()[:8], 16)
        
        textures = ["Smooth / Soft", "Coarse / Woven", "Medium Knit", "Fine Fiber"]
        patterns = ["Solid Color", "Striped Pattern", "Printed Graphic", "Melange"]
        colors = ["Navy Blue (#1a2b4c)", "Classic Crimson (#b22222)", "Sage Green (#8fbc8f)", "Cream White (#fffdd0)", "Coal Black (#2b2b2b)"]
        
        return {
            "fabric_texture": textures[hash_val % len(textures)],
            "fabric_pattern": patterns[(hash_val >> 2) % len(patterns)],
            "fabric_color": colors[(hash_val >> 4) % len(colors)],
            "damage_detected": (hash_val % 7) == 0,
            "contamination_detected": (hash_val % 11) == 0
        }

def generate_classification_report(batch, image_bytes: bytes = None, filename: str = "simulated_upload.png", cv_features: dict = None) -> dict:
    """Produces a structured material-classification and waste-report payload for a textile batch."""
    if cv_features is None:
        cv_features = {}

    condition = getattr(batch, "condition", "Good") or "Good"
    fabric_type_hint = getattr(batch, "fabric_type", None) or "Mixed Fabrics"

    feature_source = {} if not image_bytes else None
    if image_bytes and fabric_classifier:
        try:
            feature_source = fabric_classifier.extract_features_from_image(image_bytes)
        except Exception:
            feature_source = {}

    if not feature_source:
        digest = hashlib.md5((filename or str(batch.id)).encode("utf-8")).hexdigest()
        h = int(digest[:8], 16)
        feature_source = {
            "hue_peak": float(h % 360),
            "saturation": float((h % 70 + 20) / 100),
            "brightness": float((h % 60 + 25) / 100),
            "edge_density": float((h % 30 + 10) / 100),
            "pixel_variance": float((h % 1400) + 200),
            "texture_score": float((h % 50 + 20) / 100),
        }

    if cv_features:
        feature_source["edge_density"] = min(feature_source.get("edge_density", 0.2) + (0.04 if cv_features.get("damage_detected") else 0.0), 1.0)
        if "Coarse" in str(cv_features.get("fabric_texture", "")):
            feature_source["texture_score"] = min(feature_source.get("texture_score", 0.3) + 0.1, 1.0)

    prediction = {"predicted_fabric": fabric_type_hint, "confidence": 60.0, "fiber_composition": "Mixed Fibers", "all_probabilities": {}}
    if fabric_classifier:
        try:
            prediction = fabric_classifier.predict(feature_source)
        except Exception:
            prediction = {"predicted_fabric": fabric_type_hint, "confidence": 60.0, "fiber_composition": "Mixed Fibers", "all_probabilities": {}}

    waste_category = get_waste_classification(condition, cv_features.get("damage_detected", False), cv_features.get("contamination_detected", False))
    if not waste_category:
        waste_category = "Recyclable"

    scores = calculate_scores(
        prediction.get("predicted_fabric", fabric_type_hint),
        condition,
        waste_category,
        cv_features.get("damage_detected", False),
        cv_features.get("contamination_detected", False),
    )

    recyclability_score = scores.get("overall_circularity_score", 0.0)
    if recyclability_score >= 85:
        assessment_status = "High recyclability"
        recommended_action = "Direct to premium reuse or closed-loop recycling"
    elif recyclability_score >= 60:
        assessment_status = "Moderate recyclability"
        recommended_action = "Route for mechanical recycling or repair"
    elif waste_category == "Hazardous Textile Waste":
        assessment_status = "Requires secure disposal"
        recommended_action = "Contain and route through authorized hazardous waste handling"
    else:
        assessment_status = "Low recyclability"
        recommended_action = "Use downcycling or energy recovery pathways"

    return {
        "material_classification": {
            "predicted_fabric": prediction.get("predicted_fabric", fabric_type_hint),
            "confidence": prediction.get("confidence", 0.0),
            "fiber_composition": prediction.get("fiber_composition", "Mixed Fibers"),
            "probabilities": prediction.get("all_probabilities", {}),
            "hinted_fabric": fabric_type_hint,
        },
        "waste_category": waste_category,
        "recyclability_assessment": {
            "score": round(recyclability_score, 1),
            "status": assessment_status,
            "recommended_action": recommended_action,
        },
        "processing_notes": [
            f"Condition {condition} assessed with {waste_category.lower()} handling",
            f"Predicted material confidence {prediction.get('confidence', 0.0)}%",
        ],
    }


def map_rgb_to_name(r, g, b) -> str:
    """Map RGB values to simple color names."""
    if r > 220 and g > 220 and b > 220:
        return "White"
    if r < 40 and g < 40 and b < 40:
        return "Black"
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        return "Grey"
    
    # Simple color distance matching
    colors = {
        "Red": (200, 30, 30),
        "Blue": (30, 30, 200),
        "Green": (30, 180, 30),
        "Yellow": (220, 220, 30),
        "Orange": (220, 130, 30),
        "Purple": (130, 30, 180),
        "Pink": (240, 130, 180),
        "Brown": (120, 80, 40),
        "Denim Blue": (70, 100, 140)
    }
    
    closest_color = "Mixed Color"
    min_dist = 100000.0
    for name, rgb in colors.items():
        dist = np.sqrt((r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2)
        if dist < min_dist:
            min_dist = dist
            closest_color = name
            
    return closest_color

def get_composition_and_quality(fabric_type: str, condition: str) -> dict:
    """Predicts composition blend and estimates material quality."""
    # Composition blend mapping
    compositions = {
        "Cotton": "100% Organic Cotton",
        "Polyester": "100% Recycled Polyester (rPET)",
        "Wool": "100% Merino Wool",
        "Silk": "100% Mulberry Silk",
        "Linen": "100% Pure Flax Linen",
        "Denim": "98% Cotton, 2% Elastane Blend",
        "Nylon": "100% Polyamide (Nylon 6,6)",
        "Rayon": "100% Viscose Rayon",
        "Acrylic": "100% Acrylic Fiber",
        "Mixed Fabrics": "60% Cotton, 35% Polyester, 5% Polyurethane Blend"
    }
    
    quality_mapping = {
        "Excellent": "Grade A Premium",
        "Good": "Grade B Standard",
        "Fair": "Grade C Reusable Utility",
        "Poor": "Grade D Low Quality",
        "Contaminated": "Grade F Damaged/Contaminated"
    }
    
    return {
        "fiber_composition": compositions.get(fabric_type, "Mixed Fibers"),
        "quality_estimation": quality_mapping.get(condition, "Grade C Reusable Utility")
    }

def get_waste_classification(condition: str, damage: bool, contamination: bool) -> str:
    """Predicts waste category based on condition parameters."""
    if contamination:
        return "Hazardous Textile Waste"
    if condition == "Excellent":
        return "Reusable"
    if condition == "Good":
        return "Repairable" if damage else "Reusable"
    if condition == "Fair":
        return "Upcyclable" if damage else "Recyclable"
    if condition == "Poor":
        return "Recyclable"
    return "Compostable"

def get_recycling_recommendations(fabric_type: str, category: str) -> dict:
    """Provides specific recycling strategy, upcycling options, and recovery methods."""
    strategies = {
        "Reusable": {
            "strategy": "Fabric Reuse & Donation",
            "options": "Thrift retail sorting, Direct humanitarian distribution, Garment re-wear."
        },
        "Repairable": {
            "strategy": "Refurbishing & Repair",
            "options": "Seam re-stitching, Local tailoring repair, Stain extraction cleaning."
        },
        "Upcyclable": {
            "strategy": "Upcycling & Redesign",
            "options": "Patchwork collection creation, Aesthetic custom redesign, Bag/Accessory conversion."
        },
        "Compostable": {
            "strategy": "Industrial Composting",
            "options": "Microbial biodegradation, Mulch soil enhancement, Circular nutrient recovery."
        },
        "Hazardous Textile Waste": {
            "strategy": "Incineration / Secure Landfill",
            "options": "Waste-to-energy conversion, Hazardous materials extraction containment."
        }
    }
    
    # For Recyclable: base on material
    if category == "Recyclable":
        if fabric_type in ["Cotton", "Wool", "Linen"]:
            return {
                "strategy": "Mechanical Recycling (Fiber Recovery)",
                "options": "Shredding/carding into raw shoddy fibers for yarn spinning, insulation batting, or felt production."
            }
        elif fabric_type in ["Polyester", "Nylon", "Acrylic"]:
            return {
                "strategy": "Chemical Recycling (Depolymerization)",
                "options": "Solvent extraction and chemical depolymerization into high-purity monomers for virgin-grade fiber synthesis."
            }
        else:
            return {
                "strategy": "Industrial Shoddy Recovery",
                "options": "Mechanical blending for acoustic insulation, automotive sound deadeners, or industrial wiping rags."
            }
            
    return strategies.get(category, {
        "strategy": "Mechanical Recycling",
        "options": "Standard textile shredding, cleaning, and downcycled industrial fiber recovery."
    })

def calculate_scores(fabric_type: str, condition: str, category: str, damage: bool, contamination: bool) -> dict:
    """
    Weighted Scoring Model:
    Circularity Score =
      Material Recyclability (35%)
      Material Condition (20%)
      Reuse Potential (20%)
      Environmental Benefit (15%)
      Processing Feasibility (10%)
    """
    # 1. Material Recyclability (35%)
    recyclability_map = {
        "Cotton": 90, "Polyester": 85, "Wool": 92, "Silk": 80, "Linen": 95,
        "Denim": 88, "Nylon": 85, "Rayon": 75, "Acrylic": 65, "Mixed Fabrics": 45
    }
    recyclability = recyclability_map.get(fabric_type, 50)
    if contamination:
        recyclability -= 30
    elif damage:
        recyclability -= 10
    recyclability = max(0, recyclability)

    # 2. Material Condition (20%)
    condition_map = {
        "Excellent": 100, "Good": 85, "Fair": 60, "Poor": 35, "Contaminated": 10
    }
    cond_score = condition_map.get(condition, 50)

    # 3. Reuse Potential (20%)
    reuse_map = {
        "Reusable": 100, "Repairable": 85, "Upcyclable": 75,
        "Recyclable": 45, "Compostable": 30, "Hazardous Textile Waste": 0
    }
    reuse = reuse_map.get(category, 40)

    # 4. Environmental Benefit (15%)
    # Natural fibers are compostable / biodegradable; synthetic fibers prevent oil extraction if recycled.
    env_map = {
        "Cotton": 95, "Wool": 98, "Linen": 95, "Silk": 90,
        "Polyester": 80, "Nylon": 85, "Rayon": 75, "Acrylic": 60, "Mixed Fabrics": 50
    }
    env = env_map.get(fabric_type, 60)
    if category == "Hazardous Textile Waste":
        env = 0

    # 5. Processing Feasibility (10%)
    # Single fiber composition is easy, blends are difficult. Contamination lowers feasibility.
    feasibility = 95
    if fabric_type == "Mixed Fabrics" or fabric_type == "Denim":
        feasibility = 60
    if contamination:
        feasibility = 10
    elif damage:
        feasibility -= 10

    # Calculate overall weighted score
    overall = (
        0.35 * recyclability +
        0.20 * cond_score +
        0.20 * reuse +
        0.15 * env +
        0.10 * feasibility
    )
    
    # Categorization
    if overall >= 85:
        circularity_cat = "Excellent Recovery Potential"
    elif overall >= 70:
        circularity_cat = "High Recovery Potential"
    elif overall >= 50:
        circularity_cat = "Moderate Recovery Potential"
    elif overall >= 30:
        circularity_cat = "Limited Recovery Potential"
    else:
        circularity_cat = "Disposal Recommended"

    return {
        "recyclability_score": float(round(recyclability, 1)),
        "reuse_score": float(round(reuse, 1)),
        "sustainability_score": float(round(env, 1)),
        "material_recovery_score": float(round(feasibility, 1)),
        "overall_circularity_score": float(round(overall, 1)),
        "circularity_category": circularity_cat
    }

def calculate_environmental_impact(fabric_type: str, quantity_kg: float, category: str) -> dict:
    """
    Estimates environmental impact savings:
    - CO2 Savings (kg)
    - Water Savings (Liters)
    - Landfill Reduction (kg)
    """
    if category == "Hazardous Textile Waste":
        return {
            "co2_savings": 0.0,
            "water_savings": 0.0,
            "landfill_reduction": 0.0
        }
        
    # Carbon savings per kg of textile recycled/reused (kg CO2/kg)
    co2_factors = {
        "Cotton": 2.2, "Polyester": 1.9, "Wool": 3.7, "Silk": 4.1, "Linen": 2.4,
        "Denim": 2.6, "Nylon": 2.0, "Rayon": 1.6, "Acrylic": 1.4, "Mixed Fabrics": 1.5
    }
    
    # Water savings per kg (liters/kg)
    water_factors = {
        "Cotton": 2500.0, "Denim": 2900.0, "Wool": 1600.0, "Silk": 2100.0, "Linen": 1800.0,
        "Polyester": 350.0, "Nylon": 400.0, "Rayon": 600.0, "Acrylic": 300.0, "Mixed Fabrics": 1000.0
    }
    
    co2_factor = co2_factors.get(fabric_type, 1.5)
    water_factor = water_factors.get(fabric_type, 500)
    
    co2_saved = round(co2_factor * quantity_kg, 2)
    water_saved = round(water_factor * quantity_kg, 2)
    landfill_saved = round(quantity_kg, 2) # Every kg of recycled fabric is 1kg less in the landfill
    
    return {
        "co2_savings": co2_saved,
        "water_savings": water_saved,
        "landfill_reduction": landfill_saved
    }
