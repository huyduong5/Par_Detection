# So sánh C0–C4 trên 2 ảnh

Ảnh: `part17_20231212152250_01165_003.jpg` · `part16_20231212032414_03699_005.jpg`

## img01 — `part17_20231212152250_01165_003.jpg`

GT+ (7): Adult, Normal, Front, Short sleeves, Trousers, Shoulder bag, Walking

| Variant | Acc | Prec | Rec | F1 | #Pred+ | TP | FP | FN |
|---------|-----|------|-----|----|--------|----|----|----|
| C0 | 0.8947 | 0.5385 | 1.0000 | 0.7000 | 13 | 7 | 6 | 0 |
| C1 | 0.8947 | 0.5385 | 1.0000 | 0.7000 | 13 | 7 | 6 | 0 |
| C2 | 0.8947 | 0.5385 | 1.0000 | 0.7000 | 13 | 7 | 6 | 0 |
| C3 | 0.9123 | 0.5833 | 1.0000 | 0.7368 | 12 | 7 | 5 | 0 |
| C4 | 0.9123 | 0.5833 | 1.0000 | 0.7368 | 12 | 7 | 5 | 0 |

### Attr-level (GT+ và mọi pred khác nhau giữa các C)

| Attr | GT | C0 | C1 | C2 | C3 | C4 |
|------|----|------|------|------|------|------|
| Adult | 1 | 0.86 | 0.88 | 0.91 | 0.87 | 0.86 |
| Normal | 1 | 0.88 | 0.88 | 0.86 | 0.84 | 0.87 |
| Black hair | 0 | **0.62** | **0.65** | **0.65** | **0.72** | **0.73** |
| Front | 1 | 0.59 | 0.60 | 0.70 | 0.80 | 0.76 |
| Back | 0 | **0.52** | **0.52** | **0.57** | 0.39 | **0.50** |
| Short sleeves | 1 | 0.92 | 0.92 | 0.95 | 0.92 | 0.92 |
| Trousers | 1 | 0.61 | 0.57 | 0.65 | 0.62 | 0.67 |
| Shorts | 0 | **0.62** | **0.62** | **0.64** | **0.51** | 0.44 |
| Dress | 0 | **0.60** | **0.61** | **0.65** | **0.68** | **0.63** |
| Casual shoes | 0 | **0.55** | **0.55** | **0.55** | **0.63** | **0.59** |
| Sandals | 0 | **0.72** | **0.70** | **0.78** | **0.75** | **0.76** |
| Shoulder bag | 1 | 0.59 | 0.64 | 0.73 | 0.76 | 0.73 |
| Walking | 1 | 0.74 | 0.83 | 0.80 | 0.83 | 0.85 |

### C3 visibility (top-low / top-high)

Lowest vis: Leather shoes=0.23, Other shoes=0.23, Boots=0.26, Sandals=0.27, Casual shoes=0.27, Standing=0.34, Others=0.36, Riding a skateboard=0.45
Highest vis: Suitcase=0.84, Short skirt=0.84, Short sleeves=0.85, Shirt=0.85, Jeans=0.85, Coat=0.85, Backpack=0.85, Handbag=0.86

### C4 visibility (top-low / top-high)

Lowest vis: Boots=0.18, Other shoes=0.18, Leather shoes=0.19, Casual shoes=0.22, Sandals=0.22, Others=0.36, Standing=0.36, Riding a skateboard=0.37
Highest vis: Paper bag=0.85, Short sleeves=0.85, Shirt=0.85, Long skirt=0.85, Backpack=0.85, Jeans=0.85, Short skirt=0.85, Handbag=0.86

## img02 — `part16_20231212032414_03699_005.jpg`

GT+ (10): Adult, Normal, Hat, Gloves, Front, Long sleeves, Cotton-padded coat, Trousers, Casual shoes, Standing

| Variant | Acc | Prec | Rec | F1 | #Pred+ | TP | FP | FN |
|---------|-----|------|-----|----|--------|----|----|----|
| C0 | 0.9825 | 0.9091 | 1.0000 | 0.9524 | 11 | 10 | 1 | 0 |
| C1 | 0.9825 | 0.9091 | 1.0000 | 0.9524 | 11 | 10 | 1 | 0 |
| C2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 10 | 0 | 0 |
| C3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 10 | 0 | 0 |
| C4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 10 | 0 | 0 |

### Attr-level (GT+ và mọi pred khác nhau giữa các C)

| Attr | GT | C0 | C1 | C2 | C3 | C4 |
|------|----|------|------|------|------|------|
| Adult | 1 | 0.84 | 0.81 | 0.86 | 0.77 | 0.76 |
| Normal | 1 | 0.91 | 0.90 | 0.94 | 0.97 | 0.94 |
| Hat | 1 | 0.97 | 0.95 | 0.98 | 0.98 | 0.98 |
| Gloves | 1 | 0.86 | 0.81 | 0.94 | 0.97 | 0.96 |
| Front | 1 | 0.95 | 0.94 | 0.88 | 0.91 | 0.88 |
| Long sleeves | 1 | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 |
| Cotton-padded coat | 1 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| Trousers | 1 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
| Casual shoes | 1 | 0.64 | 0.75 | 0.86 | 0.85 | 0.86 |
| Boots | 0 | **0.75** | **0.78** | 0.28 | 0.38 | 0.44 |
| Standing | 1 | 0.82 | 0.80 | 0.88 | 0.87 | 0.85 |

### C3 visibility (top-low / top-high)

Lowest vis: Boots=0.15, Casual shoes=0.20, Other shoes=0.22, Sandals=0.22, Others=0.37, Standing=0.40, Leather shoes=0.43, Walking=0.48
Highest vis: Paper bag=0.86, Suit=0.86, Plastic bag=0.86, Long skirt=0.86, Backpack=0.86, Handbag=0.87, Jeans=0.87, Dress=0.88

### C4 visibility (top-low / top-high)

Lowest vis: Boots=0.14, Other shoes=0.18, Casual shoes=0.18, Sandals=0.20, Leather shoes=0.24, Standing=0.31, Walking=0.38, Riding a skateboard=0.49
Highest vis: Gloves=0.86, Jacket=0.86, Dress=0.86, Paper bag=0.86, Handbag=0.87, Suit=0.87, Chef uniform=0.87, Coat=0.87
