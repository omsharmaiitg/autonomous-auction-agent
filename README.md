# Automated Bidding Model for Used Car Auctions
### Coding Club Recruitment — Task 1

## Overview
An autonomous bidding agent that participates in live used-car auctions. The agent predicts vehicle selling prices using a tuned XGBoost model and places bids using a deterministic sigmoid-proportional bidding strategy.

## Project Structure
```
├── agent_OmSharma.py              # Live auction agent
├── model_OmSharma.pkl             # Trained XGBoost model
├── encoders_OmSharma.pkl          # Fitted LabelEncoders
├── analysis_OmSharma_Phase1.ipynb # EDA notebook
├── analysis_OmSharma_Phase2.ipynb # Model training notebook
├── report_OmSharma.pdf            # Technical report
├── requirements.txt               # Dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
`car_auction_train.csv` — historical used-car auction records (US)

| Column | Description |
|---|---|
| year | Manufacture year |
| make | Brand |
| model | Model name |
| trim | Variant |
| body | Body type |
| transmission | Automatic / Manual |
| state | US state |
| condition | Numeric condition rating |
| odometer | Mileage |
| color | Exterior colour |
| interior | Interior colour |
| sellingprice | Target — final hammer price |

## Model Pipeline

### Feature Engineering
| Feature | Formula |
|---|---|
| age_years | 2016 - year |
| usage_intensity | odometer / age_years |
| odo_per_condition | odometer / condition |
| cond_x_year | condition × year |

### Model
- **Algorithm:** XGBoost Regressor
- **Target transform:** log1p(sellingprice) → expm1 at inference
- **Tuning:** RandomizedSearchCV, 50 iterations, 3-fold CV
- **Val RMSE:** $1,620 (vs $1,890 default — 14.3% improvement)

## Bidding Strategy

### Formula
```
V_adj    = predicted_value × 0.95
σ(r)     = 1 / (1 + exp(-0.35 × (r - 5)))
ceiling  = V_adj × C(bankroll) × (1 + 0.10 × σ(r))
max_bid  = min(ceiling, V_adj, bankroll)
f(r)     = 0.40 + 0.50 × σ(r)
next_bid = current_bid + max(f(r) × gap, $50)
```

### Bankroll-Adaptive Ceiling C(B)
| Bankroll | Ceiling |
|---|---|
| ≥ 60% of initial | 0.88 |
| 30–60% | 0.70–0.80 (interpolated) |
| < 30% | 0.70 |

### Key Rules
- Win = positive profit only
- Folding (return 0.0) is never penalised
- Hard cap at V_adj guarantees profit on every win

## Agent Interface
```python
agent = LiveAuctionAgent()        # loads model + encoders
agent.analyze_item(item_dict)     # predicts price, resets round
bid = agent.place_bid(highest)    # returns next bid or 0.0 to fold
agent.auction_result(won, winning_bid, actual_price, bankroll)
```

## Results
| Criterion | Result |
|---|---|
| Prediction RMSE | $1,620 |
| Tuning improvement | 14.3% over default |
| Profit guarantee | Structural (V_adj hard cap) |
| Starting bankroll | $500,000 |
