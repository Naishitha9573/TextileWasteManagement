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
