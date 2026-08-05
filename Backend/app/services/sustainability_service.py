import logging
from typing import Any, Dict, Optional

from app.core.sustainability_constants import (
    CARBON_FACTORS,
    ENERGY_FACTORS,
    WATER_FACTORS,
)
from app.services.recommendation_engine import RecommendationEngine
from app.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)


class SustainabilityService:
    def __init__(self, scoring_service: Optional[ScoringService] = None, recommendation_engine: Optional[RecommendationEngine] = None) -> None:
        self.scoring_service = scoring_service or ScoringService()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()

    def analyze_material(
        self,
        material: str,
        condition: str,
        quantity: float,
        damage: bool = False,
        contamination: bool = False,
    ) -> Dict[str, Any]:
        waste_category = self._classify_waste(condition, damage, contamination)
        scores = self.scoring_service.calculate_scores(material, condition, waste_category, damage, contamination)
        recommendation = self.recommendation_engine.build_recommendation(
            material=material,
            condition=condition,
            waste_category=waste_category,
            damage=damage,
            contamination=contamination,
            recyclability_score=scores["recyclability_score"],
            reuse_score=scores["reuse_score"],
        )

        environmental_impact = self._estimate_environmental_impact(material, quantity, waste_category)
        logger.info("Sustainability analysis completed", extra={"material": material, "condition": condition, "waste_category": waste_category})

        return {
            "material": material,
            "condition": condition,
            "waste_category": waste_category,
            "scores": scores,
            "recommendation": recommendation,
            "environmental_impact": environmental_impact,
        }

    def _classify_waste(self, condition: str, damage: bool, contamination: bool) -> str:
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

    def _estimate_environmental_impact(self, material: str, quantity: float, waste_category: str) -> Dict[str, Any]:
        carbon_factor = CARBON_FACTORS.get(material, 3.0)
        water_factor = WATER_FACTORS.get(material, 1500)
        energy_factor = ENERGY_FACTORS.get(material, 0.8)

        if waste_category == "Hazardous Textile Waste":
            return {
                "co2_savings": 0.0,
                "water_savings": 0.0,
                "energy_savings": 0.0,
                "landfill_reduction": 0.0,
                "resource_recovery": 0.0,
            }

        co2_savings = round(quantity * carbon_factor, 2)
        water_savings = round(quantity * water_factor, 2)
        energy_savings = round(quantity * energy_factor, 2)
        landfill_reduction = round(quantity * 1.0, 2)
        resource_recovery = round(quantity * 0.7, 2)

        return {
            "co2_savings": co2_savings,
            "water_savings": water_savings,
            "energy_savings": energy_savings,
            "landfill_reduction": landfill_reduction,
            "resource_recovery": resource_recovery,
        }
