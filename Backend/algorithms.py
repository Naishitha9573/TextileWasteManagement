import io
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

from app.services.scoring_service import calculate_circularity_score
from app.services.color_analysis import analyze_image_color_bytes
from app.services.environmental_impact_service import calculate_environmental_impact as calculate_configured_impact


def _log(scope: str, message: str) -> None:
    print(f"[{scope}] {message}")


def _load_waste_rules() -> dict:
    config_path = Path(__file__).resolve().parent / "config" / "waste_rules.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def analyze_image(image_bytes: bytes, filename: str) -> dict:
    """
    Parses an uploaded image using PIL and numpy to extract features:
    - Dominant/Average color
    - Edge density (for texture and pattern)
    - Value statistics for texture and pattern analysis
    If image loading fails, returns unknown visual features. Filename-derived values are not evidence.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        color_result = analyze_image_color_bytes(image_bytes)
        image.thumbnail((250, 250))
        img_np = np.array(image)
        
        # 1. Color analysis
        avg_color = img_np.mean(axis=(0, 1))  # [R, G, B]
        r, g, b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        color_hex = f"#{r:02x}{g:02x}{b:02x}"
        
        color_name, color_confidence = map_rgb_to_name(r, g, b)

        # 2. Texture & Pattern analysis
        gray = image.convert("L")
        gray_np = np.array(gray)
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

        return {
            "fabric_texture": texture,
            "fabric_pattern": pattern,
            "color_name": color_name,
            "color_hex": color_hex,
            "color_confidence": color_confidence,
            "fabric_color": f"{color_name} ({color_hex})",
            "color": color_result["color"],
            "dominant_colors": color_result["dominant_colors"],
        }

    except Exception:
        return {
            "fabric_texture": "Unknown",
            "fabric_pattern": "Unknown",
            "color_name": "Unknown",
            "color_hex": None,
            "color_confidence": 0.0,
            "fabric_color": "Unknown",
        }


def generate_classification_report(batch, image_bytes: bytes = None, filename: str = "simulated_upload.png", cv_features: dict = None) -> dict:
    """
    Produces a structured material-classification and waste-report payload for a textile batch.

    ML material results come ONLY from model inference. When no image is provided or the
    model is unavailable, `predicted_fabric` carries only the user-declared batch hint
    (clearly labelled via `source`), never a fabricated ML output.
    """
    if cv_features is None:
        cv_features = {}

    condition = getattr(batch, "condition", "Good") or "Good"
    fabric_type_hint = getattr(batch, "fabric_type", None) or "Mixed Fabrics"

    # Runtime model state (loaded once, cached by the singleton service).
    try:
        from app.services.fabric_classifier import get_fabric_classifier
        _service = get_fabric_classifier()
        runtime_model_status = "AVAILABLE" if _service.is_ready else "UNAVAILABLE"
    except Exception:
        _service = None
        runtime_model_status = "UNAVAILABLE"

    material_source = "NO_IMAGE"
    prediction = {
        "predicted_fabric": fabric_type_hint,
        "ml_material": None,
        "confidence": None,
        "confidence_percent": None,
        "fiber_composition": get_composition_and_quality(fabric_type_hint, condition).get("fiber_composition", "Mixed Fibers"),
        "all_probabilities": {},
        "probabilities": {},
        "probability_ratios": None,
        "low_confidence": False,
        "warning": None,
        "model_version": None,
        "model_available": False,
        "model_status": runtime_model_status,
        "model_name": (_service.health().get("model_name") if _service else None),
    }

    if image_bytes:
        try:
            from app.services.fabric_classifier import ModelNotReadyError, get_fabric_classifier
            classifier = get_fabric_classifier()
            image = Image.open(io.BytesIO(image_bytes))
            prediction_result = classifier.predict(image)

            if prediction_result.get("success"):
                material_source = "MODEL"
                predicted_material = prediction_result["prediction"]["class_name"]
                confidence_value = prediction_result["prediction"]["confidence"] * 100
                prediction = {
                    "predicted_fabric": predicted_material,
                    "ml_material": predicted_material,
                    "confidence": confidence_value,
                    "confidence_percent": confidence_value,
                    "confidence_ratio": confidence_value / 100,
                    "confidence_status": "AVAILABLE",
                    "manual_review_required": False,
                    "fiber_composition": "Unknown / not laboratory verified" if predicted_material == "UNKNOWN / UNSUPPORTED" else get_composition_and_quality(predicted_material, condition).get("fiber_composition", "Unknown / not laboratory verified"),
                    "all_probabilities": {name: value * 100 for name, value in prediction_result["probabilities"].items()},
                    "probabilities": {name: value * 100 for name, value in prediction_result["probabilities"].items()},
                    "probability_ratios": prediction_result["probabilities"],
                    "model_name": "EfficientNet-B0",
                    "num_classes": len(classifier.class_names),
                    "supported_classes": list(classifier.class_names),
                    "model_available": True,
                    "model_status": "AVAILABLE",
                }
            else:
                material_source = "MODEL_NOT_AVAILABLE"
                prediction["source_note"] = (
                    "ML classification could not be performed; "
                    "'predicted_fabric' is the user-declared batch hint only."
                )
                prediction["warning"] = prediction_result.get(
                    "message", "ML model unavailable — material classification not performed."
                )
                prediction["error"] = prediction_result.get("message")
                prediction["model_status"] = prediction_result.get("model_status") or runtime_model_status
        except ModelNotReadyError as exc:
            material_source = "MODEL_NOT_AVAILABLE"
            prediction["warning"] = "Model inference unavailable — material classification not performed."
            prediction["error"] = "EfficientNet-B0 model is not ready."
            print(f"[FABRIC MODEL] NOT_READY: {exc}")
            prediction["model_status"] = runtime_model_status
        except Exception as exc:
            material_source = "MODEL_NOT_AVAILABLE"
            prediction["warning"] = "Model inference unavailable — material classification not performed."
            prediction["error"] = "Internal error during inference."
            print(f"[FABRIC MODEL] ERROR: inference raised: {exc}")
            prediction["model_status"] = runtime_model_status
    else:
        prediction["warning"] = "No image provided — material from manual batch hint only."

    damage_detected = bool(cv_features.get("damage_detected", False))
    contamination_detected = bool(cv_features.get("contamination_detected", False))
    waste_category = get_waste_classification(condition, damage_detected, contamination_detected)
    waste_source = "RULE_ENGINE"
    if not waste_category:
        waste_category = "Recyclable"

    scores = calculate_scores(
        prediction.get("predicted_fabric", fabric_type_hint),
        condition,
        waste_category,
        damage_detected,
        contamination_detected,
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

    confidence_display = prediction.get("confidence")
    confidence_note_text = (
        f"{confidence_display}%" if confidence_display is not None else "Not available"
    )
    _log("RULE_ENGINE", f"Recovery potential: {round(recyclability_score, 1)}%")

    processing_notes = [
        f"Material source: {material_source}",
        f"Waste category source: {waste_source}",
        f"Condition {condition} → {waste_category}",
        f"Material confidence: {confidence_note_text}",
    ]
    if material_source != "MODEL":
        processing_notes.append(
            "Recovery assessment uses the user-declared fabric hint; no ML material classification was applied."
        )
    if prediction.get("warning"):
        processing_notes.append(prediction["warning"])
    model_classes = prediction.get("supported_classes") or (_service.class_names if _service else [])
    return {
        "material_classification": {
            "predicted_fabric": prediction.get("predicted_fabric") or "UNKNOWN / UNSUPPORTED",
            "ml_material": prediction.get("ml_material"),
            "model_available": prediction.get("model_available", False),
            "confidence": prediction.get("confidence"),
            "confidence_percent": prediction.get("confidence_percent"),
            "confidence_ratio": prediction.get("confidence_ratio"),
            "confidence_status": prediction.get("confidence_status"),
            "confidence_threshold": prediction.get("confidence_threshold"),
            "manual_review_required": prediction.get("manual_review_required", False),
            "fiber_composition": prediction.get("fiber_composition", "Mixed Fibers"),
            "probabilities": prediction.get("probabilities", {}),
            "probability_ratios": prediction.get("probability_ratios"),
            "hinted_fabric": fabric_type_hint,
            "source": material_source,
            "model_version": prediction.get("model_version"),
            "model_name": prediction.get("model_name"),
            "num_classes": prediction.get("num_classes"),
            "current_model_scope": prediction.get("current_model_scope") or (f"{len(model_classes)} CLASS MODEL" if model_classes else None),
            "supported_classes": prediction.get("supported_classes", []),
            "model_accuracy": prediction.get("model_accuracy"),
            "model_test_accuracy": prediction.get("model_test_accuracy"),
            "model_status": prediction.get("model_status", runtime_model_status),
            "error": prediction.get("error"),
            "confidence_note": "Confidence represents the model's classification score among supported classes and is not laboratory-verified fiber composition.",
            "unknown_or_unsupported": prediction.get("unknown_or_unsupported", False),
            "low_confidence": prediction.get("low_confidence", False),
        },
        "waste_category": waste_category,
        "waste_category_source": waste_source,
        "recyclability_assessment": {
            "score": round(recyclability_score, 1),
            "status": assessment_status,
            "recommended_action": recommended_action,
            "source": "RULE_ENGINE",
        },
        "processing_notes": processing_notes,
    }


def map_rgb_to_name(r, g, b) -> tuple[str, float]:
    if r > 220 and g > 220 and b > 220:
        return "White", 0.95
    if r < 40 and g < 40 and b < 40:
        return "Black", 0.95
    if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
        return "Grey", 0.9
    if r > g * 1.5 and b > g * 1.5 and b >= r * 0.9:
        return "Purple", 0.9
    
    colors = {
        "Red": (200, 30, 30),
        "Blue": (30, 30, 200),
        "Green": (30, 180, 30),
        "Yellow": (220, 220, 30),
        "Orange": (220, 130, 30),
        "Purple": (130, 30, 180),
        "Pink": (240, 130, 180),
        "Brown": (120, 80, 40),
        "Navy Blue": (30, 40, 100),
    }
    
    closest_color = "Mixed Color"
    min_dist = 100000.0
    for name, rgb in colors.items():
        dist = np.sqrt((r - rgb[0])**2 + (g - rgb[1])**2 + (b - rgb[2])**2)
        if dist < min_dist:
            min_dist = dist
            closest_color = name
            
    confidence = max(0.0, min(1.0, 1.0 - min_dist / 320.0))
    return closest_color, round(confidence, 3)


def get_composition_and_quality(fabric_type: str, condition: str) -> dict:
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
    """Predicts waste category using waste_rules.yaml if available, else standard fallback."""
    rules_cfg = _load_waste_rules()
    rules = rules_cfg.get("rules", [])

    for rule in rules:
        c_match = rule.get("condition_match")
        d_match = rule.get("damage_match")
        cont_match = rule.get("contamination_match")

        if cont_match is not None and cont_match != contamination:
            continue
        if c_match is not None and c_match != condition:
            continue
        if d_match is not None and d_match != damage:
            continue

        return rule.get("waste_category", "Recyclable")

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
    return rules_cfg.get("default_category", "Compostable")


def get_recycling_recommendations(fabric_type: str, category: str) -> dict:
    if fabric_type == "UNKNOWN / UNSUPPORTED":
        return {
            "strategy": "Manual material verification",
            "options": "Hold for textile specialist review before selecting a material-specific pathway.",
            "confidence": 0.0,
        }
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
    Weighted Scoring Model using canonical calculate_circularity_score():
      Material Recyclability: 35%
      Material Condition / Recovery: 20%
      Reuse Potential: 20%
      Environmental Benefit: 15%
      Processing Feasibility: 10%
    """
    # 1. Material Recyclability (35%)
    recyclability_map = {
        "Cotton": 90, "Polyester": 85, "Wool": 92, "Silk": 80, "Linen": 95,
        "Denim": 88, "Nylon": 85, "Rayon": 75, "Acrylic": 65, "Mixed Fabrics": 45
    }
    recyclability = float(recyclability_map.get(fabric_type, 50))
    if contamination:
        recyclability -= 30
    elif damage:
        recyclability -= 10
    recyclability = max(0.0, recyclability)

    # 2. Material Condition / Recovery (20%)
    condition_map = {
        "Excellent": 100, "Good": 85, "Fair": 60, "Poor": 35, "Contaminated": 10
    }
    cond_score = float(condition_map.get(condition, 50))

    # 3. Reuse Potential (20%)
    reuse_map = {
        "Reusable": 100, "Repairable": 85, "Upcyclable": 75,
        "Recyclable": 45, "Compostable": 30, "Hazardous Textile Waste": 0
    }
    reuse = float(reuse_map.get(category, 40))

    # 4. Environmental Benefit (15%)
    env_map = {
        "Cotton": 95, "Wool": 98, "Linen": 95, "Silk": 90,
        "Polyester": 80, "Nylon": 85, "Rayon": 75, "Acrylic": 60, "Mixed Fabrics": 50
    }
    env = float(env_map.get(fabric_type, 60))
    if category == "Hazardous Textile Waste":
        env = 0.0

    # 5. Processing Feasibility (10%)
    feasibility = 95.0
    if fabric_type == "Mixed Fabrics" or fabric_type == "Denim":
        feasibility = 60.0
    if contamination:
        feasibility = 10.0
    elif damage:
        feasibility -= 10.0

    overall = calculate_circularity_score(
        material_recyclability_score=recyclability,
        material_condition_score=cond_score,
        reuse_potential_score=reuse,
        environmental_benefit_score=env,
        processing_feasibility_score=feasibility,
    )
    
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
    impact = calculate_configured_impact(fabric_type, quantity_kg, category)
    co2_saved = impact["co2"]["value"]
    water_saved = impact["water"]["value"]
    landfill_saved = 0.0 if category == "Hazardous Textile Waste" else round(quantity_kg, 2)
    _log(
        "SUSTAINABILITY",
        f"material={fabric_type} quantity_kg={quantity_kg} co2_factor={impact['co2'].get('factor')} "
        f"water_factor={impact['water'].get('factor')} calculated_co2={co2_saved} calculated_water={water_saved}",
    )
    return {
        "co2_savings": co2_saved,
        "water_savings": water_saved,
        "landfill_reduction": landfill_saved,
        "co2_factor": impact["co2"].get("factor"),
        "water_factor": impact["water"].get("factor"),
        "factor_status": impact["calculation_status"],
        "methodology": impact["basis"],
        "environmental_impact": impact,
    }
