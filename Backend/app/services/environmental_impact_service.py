"""Centralized, transparent environmental impact estimates."""
from __future__ import annotations

from typing import Any

from app.core.sustainability_constants import (
    CARBON_FACTORS,
    FACTOR_SOURCE,
    WATER_FACTORS,
)

CO2_UNIT = "kg CO2e"
CO2_FACTOR_UNIT = "kg CO2e/kg"
WATER_UNIT = "L"
WATER_FACTOR_UNIT = "L/kg"


def _factor_metadata(material: str, factors: dict[str, float], unit: str) -> dict[str, Any]:
    value = factors.get(material)
    return {
        "material": material,
        "value": value,
        "unit": unit,
        "basis": FACTOR_SOURCE,
        "status": "ESTIMATE" if value is not None else "FACTOR_NOT_CONFIGURED",
    }


def _unavailable_result(material: str | None, quantity_kg: float | None, reason: str) -> dict[str, Any]:
    return {
        "success": False,
        "material": material,
        "material_status": "AVAILABLE" if material else "UNAVAILABLE",
        "quantity_kg": quantity_kg,
        "quantity_status": "AVAILABLE" if quantity_kg is not None else "UNAVAILABLE",
        "co2": {
            "value": None,
            "unit": CO2_UNIT,
            "status": "FACTOR_NOT_CONFIGURED" if reason == "factor" else "UNAVAILABLE",
            "factor": None,
            "factor_unit": CO2_FACTOR_UNIT,
        },
        "water": {
            "value": None,
            "unit": WATER_UNIT,
            "status": "FACTOR_NOT_CONFIGURED" if reason == "factor" else "UNAVAILABLE",
            "factor": None,
            "factor_unit": WATER_FACTOR_UNIT,
        },
        "calculation_status": "FACTOR_NOT_CONFIGURED" if reason == "factor" else "UNAVAILABLE",
        "source": "configured_project_factors",
        "basis": FACTOR_SOURCE,
        "message": {
            "material": "Material is unavailable.",
            "quantity": "Quantity is unavailable or must be greater than zero.",
            "factor": "Environmental impact factors not configured.",
        }[reason],
    }


def calculate_environmental_impact(
    material: str | None,
    quantity_kg: float | None,
    category: str | None = None,
) -> dict[str, Any]:
    """Calculate potential savings from configured factors and real inputs.

    No recovery multiplier is applied because the existing project has no
    configured environmental recovery factor in this calculation path.
    """
    if not material or material == "UNKNOWN / UNSUPPORTED":
        return _unavailable_result(material, quantity_kg, "material")
    if quantity_kg is not None and quantity_kg < 0:
        raise ValueError("quantity_kg must be non-negative")
    if quantity_kg is None or quantity_kg == 0:
        return _unavailable_result(material, quantity_kg, "quantity")

    if category == "Hazardous Textile Waste":
        return {
            "success": True,
            "material": material,
            "material_status": "AVAILABLE",
            "quantity_kg": quantity_kg,
            "quantity_status": "AVAILABLE",
            "co2": {"value": 0.0, "unit": CO2_UNIT, "status": "NOT_APPLICABLE", "factor": None, "factor_unit": CO2_FACTOR_UNIT},
            "water": {"value": 0.0, "unit": WATER_UNIT, "status": "NOT_APPLICABLE", "factor": None, "factor_unit": WATER_FACTOR_UNIT},
            "calculation_status": "NOT_APPLICABLE_HAZARDOUS",
            "source": "configured_project_factors",
            "basis": "Environmental savings are not applicable to hazardous textile waste.",
        }

    co2 = _factor_metadata(material, CARBON_FACTORS, CO2_FACTOR_UNIT)
    water = _factor_metadata(material, WATER_FACTORS, WATER_FACTOR_UNIT)
    if co2["value"] is None or water["value"] is None:
        result = _unavailable_result(material, quantity_kg, "factor")
        result["co2"] = {
            "value": round(quantity_kg * co2["value"], 2) if co2["value"] is not None else None,
            "unit": CO2_UNIT,
            "status": co2["status"],
            "factor": co2["value"],
            "factor_unit": CO2_FACTOR_UNIT,
            "basis": co2["basis"],
        }
        result["water"] = {
            "value": round(quantity_kg * water["value"], 2) if water["value"] is not None else None,
            "unit": WATER_UNIT,
            "status": water["status"],
            "factor": water["value"],
            "factor_unit": WATER_FACTOR_UNIT,
            "basis": water["basis"],
        }
        return result

    result = {
        "success": True,
        "material": material,
        "material_status": "AVAILABLE",
        "quantity_kg": quantity_kg,
        "quantity_status": "AVAILABLE",
        "co2": {
            "value": round(quantity_kg * co2["value"], 2),
            "unit": CO2_UNIT,
            "status": "ESTIMATE",
            "factor": co2["value"],
            "factor_unit": CO2_FACTOR_UNIT,
            "basis": co2["basis"],
        },
        "water": {
            "value": round(quantity_kg * water["value"], 2),
            "unit": WATER_UNIT,
            "status": "ESTIMATE",
            "factor": water["value"],
            "factor_unit": WATER_FACTOR_UNIT,
            "basis": water["basis"],
        },
        "calculation_status": "ESTIMATED",
        "source": "configured_project_factors",
        "basis": FACTOR_SOURCE,
    }
    result["co2_savings"] = result["co2"]["value"]
    result["water_savings"] = result["water"]["value"]
    return result


def calculate_co2_savings(material: str | None, quantity_kg: float | None) -> dict[str, Any]:
    return calculate_environmental_impact(material, quantity_kg)["co2"]


def calculate_water_savings(material: str | None, quantity_kg: float | None) -> dict[str, Any]:
    return calculate_environmental_impact(material, quantity_kg)["water"]
