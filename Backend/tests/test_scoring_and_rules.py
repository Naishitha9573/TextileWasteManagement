"""Unit tests for circularity scoring formula and waste rules engine."""
import pytest
from app.services.scoring_service import calculate_circularity_score, ScoringService
from algorithms import get_waste_classification, calculate_scores


def test_circularity_score_bounds_and_weights():
    # Test max score (100)
    score_max = calculate_circularity_score(100, 100, 100, 100, 100)
    assert score_max == 100.0

    # Test min score (0)
    score_min = calculate_circularity_score(0, 0, 0, 0, 0)
    assert score_min == 0.0

    # Test expected weighted sum:
    # 80*0.35 (28) + 70*0.20 (14) + 60*0.20 (12) + 90*0.15 (13.5) + 50*0.10 (5.0) = 72.5
    expected = 80 * 0.35 + 70 * 0.20 + 60 * 0.20 + 90 * 0.15 + 50 * 0.10
    score_calc = calculate_circularity_score(80, 70, 60, 90, 50)
    assert score_calc == round(expected, 1)


def test_scoring_service_calculate_scores():
    svc = ScoringService()
    scores = svc.calculate_scores(
        material="Cotton",
        condition="Good",
        waste_category="Reusable",
        damage=False,
        contamination=False,
    )
    assert "circular_economy_score" in scores
    assert "overall_sustainability_score" in scores
    assert 0 <= scores["overall_sustainability_score"] <= 100
    assert scores["sustainability_rating"] in ["Excellent", "Good", "Needs Attention"]


def test_waste_rules_classification():
    # Contaminated -> Hazardous
    assert get_waste_classification("Good", False, True) == "Hazardous Textile Waste"

    # Excellent, undamaged -> Reusable
    assert get_waste_classification("Excellent", False, False) == "Reusable"

    # Good, damaged -> Repairable
    assert get_waste_classification("Good", True, False) == "Repairable"

    # Good, undamaged -> Reusable
    assert get_waste_classification("Good", False, False) == "Reusable"

    # Fair, damaged -> Upcyclable
    assert get_waste_classification("Fair", True, False) == "Upcyclable"

    # Fair, undamaged -> Recyclable
    assert get_waste_classification("Fair", False, False) == "Recyclable"

    # Poor -> Recyclable
    assert get_waste_classification("Poor", False, False) == "Recyclable"
