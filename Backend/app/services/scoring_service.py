from typing import Dict, Any

from app.core.sustainability_constants import CIRCULARITY_WEIGHTS, RECOMMENDATION_THRESHOLDS


class ScoringService:
    def __init__(self) -> None:
        self.weights = CIRCULARITY_WEIGHTS
        self.thresholds = RECOMMENDATION_THRESHOLDS

    def calculate_scores(
        self,
        material: str,
        condition: str,
        waste_category: str,
        damage: bool,
        contamination: bool,
    ) -> Dict[str, Any]:
        recyclability_score = self._material_recyclability(material, contamination, damage)
        reuse_score = self._reuse_score(material, condition, damage, contamination)
        material_recovery_score = self._material_recovery_score(waste_category, contamination)
        environmental_benefit_score = self._environmental_benefit_score(material, waste_category)
        processing_feasibility_score = self._processing_feasibility_score(waste_category, damage)

        circular_economy_score = (
            recyclability_score * self.weights["material_recyclability"]
            + reuse_score * self.weights["reuse_potential"]
            + material_recovery_score * self.weights["material_recyclability"]
            + environmental_benefit_score * self.weights["environmental_benefit"]
            + processing_feasibility_score * self.weights["processing_feasibility"]
        )

        overall_sustainability_score = round(circular_economy_score, 1)
        rating = self._rating(overall_sustainability_score)

        return {
            "recyclability_score": round(recyclability_score, 1),
            "reuse_score": round(reuse_score, 1),
            "material_recovery_score": round(material_recovery_score, 1),
            "circular_economy_score": round(circular_economy_score, 1),
            "environmental_benefit_score": round(environmental_benefit_score, 1),
            "overall_sustainability_score": overall_sustainability_score,
            "sustainability_rating": rating,
        }

    def _material_recyclability(self, material: str, contamination: bool, damage: bool) -> float:
        base = {"Cotton": 90, "Polyester": 85, "Wool": 92, "Silk": 80, "Linen": 95, "Denim": 88, "Mixed Fabrics": 60}.get(material, 70)
        if contamination:
            base -= 30
        if damage:
            base -= 10
        return max(0, min(100, base))

    def _reuse_score(self, material: str, condition: str, damage: bool, contamination: bool) -> float:
        base = {"Excellent": 90, "Good": 80, "Fair": 65, "Poor": 50, "Contaminated": 30}.get(condition, 60)
        if contamination:
            base -= 20
        if damage:
            base -= 10
        if material in {"Cotton", "Linen", "Denim"}:
            base += 5
        return max(0, min(100, base))

    def _material_recovery_score(self, waste_category: str, contamination: bool) -> float:
        base = {"Reusable": 95, "Repairable": 80, "Upcyclable": 75, "Recyclable": 85, "Compostable": 65, "Hazardous Textile Waste": 20}.get(waste_category, 70)
        if contamination:
            base -= 10
        return max(0, min(100, base))

    def _environmental_benefit_score(self, material: str, waste_category: str) -> float:
        base = {"Cotton": 85, "Polyester": 70, "Denim": 80, "Silk": 75, "Linen": 90, "Wool": 78, "Mixed Fabrics": 65}.get(material, 70)
        if waste_category == "Hazardous Textile Waste":
            base -= 40
        return max(0, min(100, base))

    def _processing_feasibility_score(self, waste_category: str, damage: bool) -> float:
        base = {"Reusable": 95, "Repairable": 85, "Upcyclable": 75, "Recyclable": 80, "Compostable": 60, "Hazardous Textile Waste": 20}.get(waste_category, 70)
        if damage:
            base -= 10
        return max(0, min(100, base))

    def _rating(self, score: float) -> str:
        if score >= self.thresholds["high_recyclability"]:
            return "Excellent"
        if score >= self.thresholds["moderate_recyclability"]:
            return "Good"
        return "Needs Attention"
