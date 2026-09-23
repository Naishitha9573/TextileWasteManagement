from typing import Any, Dict

from app.core.sustainability_constants import CIRCULARITY_WEIGHTS, RECOMMENDATION_THRESHOLDS


def calculate_circularity_score(
    material_recyclability_score: float,
    material_condition_score: float,
    reuse_potential_score: float,
    environmental_benefit_score: float,
    processing_feasibility_score: float,
) -> float:
    """Canonical formula for circular economy score (0 to 100).
    Weights sum to 1.0 (100%):
      - Material Recyclability: 35%
      - Material Condition / Recovery: 20%
      - Reuse Potential: 20%
      - Environmental Benefit: 15%
      - Processing Feasibility: 10%
    """
    weights = CIRCULARITY_WEIGHTS
    raw_score = (
        material_recyclability_score * weights["material_recyclability"]
        + material_condition_score * weights["material_condition"]
        + reuse_potential_score * weights["reuse_potential"]
        + environmental_benefit_score * weights["environmental_benefit"]
        + processing_feasibility_score * weights["processing_feasibility"]
    )
    return max(0.0, min(100.0, round(raw_score, 1)))


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
        """
        Single authoritative scoring path.

        Delegates to algorithms.calculate_scores, which implements the canonical
        circularity formula (35/20/20/15/10) and the five recovery categories.
        Returns a key-superset so both legacy consumers and the unified
        inference response work from one implementation.
        """
        import algorithms

        base = algorithms.calculate_scores(material, condition, waste_category, damage, contamination)

        recyclability_score = base["recyclability_score"]
        reuse_score = base["reuse_score"]
        environmental_benefit_score = base["sustainability_score"]
        processing_feasibility_score = base["material_recovery_score"]
        circular_economy_score = base["overall_circularity_score"]
        category = base["circularity_category"]

        # Condition component (same derivation rules as the authoritative path).
        condition_map = {"Excellent": 100, "Good": 85, "Fair": 60, "Poor": 35, "Contaminated": 10}
        condition_score = float(condition_map.get(condition, 50))

        overall_sustainability_score = round(circular_economy_score, 1)

        return {
            # Spec-required keys
            "recyclability_score": recyclability_score,
            "condition_score": condition_score,
            "reuse_score": reuse_score,
            "environmental_score": environmental_benefit_score,
            "processing_feasibility_score": processing_feasibility_score,
            "circularity_score": circular_economy_score,
            "circularity_category": category,
            # Legacy keys preserved for backward compatibility
            "material_recovery_score": processing_feasibility_score,
            "sustainability_score": environmental_benefit_score,
            "circular_economy_score": circular_economy_score,
            "environmental_benefit_score": environmental_benefit_score,
            "overall_sustainability_score": overall_sustainability_score,
            # Legacy 3-level rating, distinct from the 5-level circularity_category
            "sustainability_rating": self._rating(overall_sustainability_score),
        }

    def _material_recyclability(self, material: str, contamination: bool, damage: bool) -> float:
        base = {"Cotton": 90, "Polyester": 85, "Wool": 92, "Silk": 80, "Linen": 95, "Denim": 88, "Mixed Fabrics": 60}.get(material, 20)
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
        base = {"Cotton": 85, "Polyester": 70, "Denim": 80, "Silk": 75, "Linen": 90, "Wool": 78, "Mixed Fabrics": 65}.get(material, 20)
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
