"""Project sustainability factors used for transparent mathematical estimates.

These are configuration values, not laboratory measurements. They represent the
project's current lifecycle-factor assumptions and are returned with calculations
so callers can distinguish estimates from measured impacts.
"""

FACTOR_SOURCE = "Project lifecycle-factor configuration; estimates require domain validation before production use."

CARBON_FACTORS = {
    "Cotton": 2.2,
    "Polyester": 1.9,
    "Wool": 3.7,
    "Silk": 4.1,
    "Linen": 2.4,
    "Denim": 2.6,
    "Nylon": 2.0,
    "Rayon": 1.6,
    "Acrylic": 1.4,
    "Mixed Fabrics": 1.5,
}

WATER_FACTORS = {
    "Cotton": 2500.0,
    "Polyester": 350.0,
    "Wool": 1600.0,
    "Silk": 2100.0,
    "Linen": 1800.0,
    "Denim": 2900.0,
    "Nylon": 400.0,
    "Rayon": 600.0,
    "Acrylic": 300.0,
    "Mixed Fabrics": 1000.0,
}

ENERGY_FACTORS = {
    "Cotton": 0.7,
    "Polyester": 1.2,
    "Denim": 0.9,
    "Silk": 1.5,
    "Linen": 0.6,
    "Wool": 1.0,
    "Mixed Fabrics": 0.8,
}

RECOVERY_PERCENTAGES = {
    "Reusable": 0.95,
    "Repairable": 0.75,
    "Upcyclable": 0.65,
    "Recyclable": 0.8,
    "Compostable": 0.5,
    "Hazardous Textile Waste": 0.1,
}

PROCESSING_DIFFICULTY = {
    "Reusable": "Low",
    "Repairable": "Medium",
    "Upcyclable": "Medium",
    "Recyclable": "Medium",
    "Compostable": "High",
    "Hazardous Textile Waste": "High",
}

CIRCULARITY_WEIGHTS = {
    "material_recyclability": 0.35,
    "material_condition": 0.20,
    "reuse_potential": 0.20,
    "environmental_benefit": 0.15,
    "processing_feasibility": 0.10,
}

RECOMMENDATION_THRESHOLDS = {
    "high_recyclability": 85,
    "moderate_recyclability": 60,
}
