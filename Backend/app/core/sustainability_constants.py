CARBON_FACTORS = {
    "Cotton": 2.4,
    "Polyester": 5.5,
    "Denim": 3.8,
    "Silk": 4.2,
    "Linen": 2.1,
    "Wool": 4.8,
    "Mixed Fabrics": 3.2,
}

WATER_FACTORS = {
    "Cotton": 2700,
    "Polyester": 120,
    "Denim": 1800,
    "Silk": 5000,
    "Linen": 2500,
    "Wool": 10000,
    "Mixed Fabrics": 1800,
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
