NUT8 = [
    'Energy(kcal)', 'Carbohydrate(g)', 'Total fat(g)', 'Protein(g)',
    'Sodium(mg)', 'Total sugar(g)', 'Saturated fatty acids(g)', 'Cholesterol(mg)'
]

NUT5 = [
    'Energy(kcal)', 'Carbohydrate(g)', 'Protein(g)',
    'Sodium(mg)', 'Total sugar(g)', 'Saturated fatty acids(g)', 'Cholesterol(mg)'
]

NUT4 = [
    'Energy(kcal)', 'Carbohydrate(g)', 'Total fat(g)', 'Protein(g)'
]

NUT3 = [
    'Carbohydrate(g)', 'Total fat(g)', 'Protein(g)'
]

NUT11 = [
    'Energy(kcal)', 'Carbohydrate(g)', 'Total fat(g)', 'Protein(g)',
    'Sodium(mg)', 'Total sugar(g)', 'Saturated fatty acids(g)', 'Cholesterol(mg)',
    'Calcium(mg)', 'Fiber(g)', 'Iron(mg)'
]

TARGET_NUT3 = [
    "Calcium(mg)", "Fiber(g)", "Iron(mg)"
]

ORDER = [
    "nut8", "nut8+binary", "nut8+score", "nut8+umap_10", "ing_pred"
]

GROUP_COLS = ["full", "first_mapped", "mapped_ratio_high", "mapped_ratio_top20_high", "strict"]