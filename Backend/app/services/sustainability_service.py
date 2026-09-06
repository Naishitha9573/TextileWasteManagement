import logging
from typing import Any, Dict, Optional

from app.core.sustainability_constants import (
    CARBON_FACTORS,
    ENERGY_FACTORS,
    FACTOR_SOURCE,
    WATER_FACTORS,
)
from app.services.recommendation_engine import RecommendationEngine
from app.services.scoring_service import ScoringService
from app.services.environmental_impact_service import calculate_environmental_impact

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
        material_confidence: float = 1.0,
        condition_confidence: float = 1.0,
        contamination_confidence: float = 1.0,
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
            material_confidence=material_confidence,
            condition_confidence=condition_confidence,
            contamination_confidence=contamination_confidence,
        )

        environmental_impact = calculate_environmental_impact(material, quantity, waste_category)
        logger.info(
            "SUSTAINABILITY material=%s quantity_kg=%s co2_factor=%s water_factor=%s calculated_co2=%s calculated_water=%s",
            material,
            quantity,
            environmental_impact.get("co2_factor"),
            environmental_impact.get("water_factor"),
            environmental_impact.get("co2_savings"),
            environmental_impact.get("water_savings"),
        )
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

