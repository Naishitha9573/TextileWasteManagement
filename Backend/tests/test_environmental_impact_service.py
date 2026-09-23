import pytest

from app.core import sustainability_constants
from app.services.environmental_impact_service import calculate_environmental_impact


def test_calculates_configured_co2_and_water_for_materials():
    for material in ("Cotton", "Polyester", "Denim"):
        result = calculate_environmental_impact(material, 2.0)
        assert result["success"] is True
        assert result["co2"]["value"] == round(2.0 * sustainability_constants.CARBON_FACTORS[material], 2)
        assert result["water"]["value"] == round(2.0 * sustainability_constants.WATER_FACTORS[material], 2)
        assert result["co2"]["status"] == "ESTIMATE"
        assert result["water"]["status"] == "ESTIMATE"


def test_rejects_negative_quantity():
    with pytest.raises(ValueError, match="non-negative"):
        calculate_environmental_impact("Cotton", -1.0)


@pytest.mark.parametrize("material, quantity", [(None, 2.0), ("", 2.0), ("Cotton", None), ("Cotton", 0.0)])
def test_missing_material_or_quantity_is_unavailable(material, quantity):
    result = calculate_environmental_impact(material, quantity)
    assert result["success"] is False
    assert result["calculation_status"] == "UNAVAILABLE"
    assert result["co2"]["value"] is None
    assert result["water"]["value"] is None


def test_unknown_material_has_no_specific_impact():
    result = calculate_environmental_impact("Unknown Textile", 2.0)
    assert result["success"] is False
    assert result["calculation_status"] == "FACTOR_NOT_CONFIGURED"
    assert result["co2"]["value"] is None
    assert result["water"]["value"] is None


def test_missing_single_factor_is_explicit(monkeypatch):
    monkeypatch.setitem(sustainability_constants.CARBON_FACTORS, "Test Textile", 1.0)
    result = calculate_environmental_impact("Test Textile", 2.0)
    assert result["success"] is False
    assert result["co2"]["value"] == 2.0
    assert result["water"]["value"] is None
    assert result["water"]["status"] == "FACTOR_NOT_CONFIGURED"
