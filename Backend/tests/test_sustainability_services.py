import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.sustainability_service import SustainabilityService


def test_sustainability_service_returns_structured_analysis():
    service = SustainabilityService()
    batch = SimpleNamespace(fabric_type="Cotton", condition="Excellent", quantity=100.0)

    result = service.analyze_material(
        material=batch.fabric_type,
        condition=batch.condition,
        quantity=batch.quantity,
        damage=False,
        contamination=False,
    )

    assert result["waste_category"] in {"Reusable", "Repairable", "Recyclable", "Upcyclable", "Compostable", "Hazardous Textile Waste"}
    assert result["scores"]["overall_sustainability_score"] >= 70
    assert result["recommendation"]["primary_recommendation"]
    assert result["environmental_impact"]["co2_savings"] >= 0


def test_sustainability_estimates_scale_with_quantity_and_material():
    service = SustainabilityService()
    cotton_one = service.analyze_material("Cotton", "Excellent", 1.0)["environmental_impact"]
    cotton_ten = service.analyze_material("Cotton", "Excellent", 10.0)["environmental_impact"]
    polyester_one = service.analyze_material("Polyester", "Excellent", 1.0)["environmental_impact"]

    assert cotton_ten["co2_savings"] == cotton_one["co2_savings"] * 10
    assert cotton_ten["water_savings"] == cotton_one["water_savings"] * 10
    assert cotton_one["co2_savings"] != polyester_one["co2_savings"]
    assert cotton_one["water_savings"] != polyester_one["water_savings"]
