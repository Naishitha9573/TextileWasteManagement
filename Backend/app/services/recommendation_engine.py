from typing import Dict, Any

from app.core.sustainability_constants import PROCESSING_DIFFICULTY, RECOVERY_PERCENTAGES


class RecommendationEngine:
    def __init__(self) -> None:
        self.processing_difficulty = PROCESSING_DIFFICULTY
        self.recovery_percentages = RECOVERY_PERCENTAGES

    def build_recommendation(
        self,
        material: str,
        condition: str,
        waste_category: str,
        damage: bool,
        contamination: bool,
        recyclability_score: float,
        reuse_score: float,
    ) -> Dict[str, Any]:
        if waste_category == "Hazardous Textile Waste":
            return {
                "primary_recommendation": "Secure disposal",
                "alternative_recommendation": "Authorized hazardous handling",
                "reason": "Contamination or hazardous content requires controlled handling.",
                "recovery_rate": 0.1,
                "difficulty": "High",
                "estimated_cost": 180.0,
                "confidence": 0.92,
            }

        if recyclability_score >= 85:
            primary = "Fiber recycling"
            alternative = "Premium reuse"
            reason = "High-quality material with strong recovery potential."
            recovery_rate = self.recovery_percentages.get(waste_category, 0.7)
            cost = 45.0
        elif reuse_score >= 70:
            primary = "Reuse and resale"
            alternative = "Repair and refurbishment"
            reason = "The material is suitable for second-life use."
            recovery_rate = 0.8
            cost = 30.0
        elif waste_category == "Upcyclable":
            primary = "Upcycling"
            alternative = "Industrial recycling"
            reason = "The item can be converted into a higher-value product."
            recovery_rate = 0.65
            cost = 55.0
        else:
            primary = "Mechanical recycling"
            alternative = "Downcycling"
            reason = "The material is still suitable for recovery with moderate processing effort."
            recovery_rate = 0.6
            cost = 40.0

        if contamination:
            primary = "Secure disposal"
            alternative = "Specialized recycling"
            reason = "Contamination reduces recovery viability."
            recovery_rate = 0.2
            cost = 120.0

        return {
            "primary_recommendation": primary,
            "alternative_recommendation": alternative,
            "reason": reason,
            "recovery_rate": round(recovery_rate, 2),
            "difficulty": self.processing_difficulty.get(waste_category, "Medium"),
            "estimated_cost": round(cost, 2),
            "confidence": round(min(0.95, 0.6 + (recyclability_score / 200)), 2),
        }
