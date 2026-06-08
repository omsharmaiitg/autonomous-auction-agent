import os
import math
import warnings
import numpy as np
import joblib

REFERENCE_YEAR = 2016

NUMERIC_COLS = [
    'year', 'condition', 'odometer',
    'usage_intensity', 'age_years',
    'odo_per_condition', 'cond_x_year',
]

CATEGORICAL_COLS = [
    'make', 'model', 'trim', 'body',
    'transmission', 'state', 'color', 'interior',
]

INITIAL_BANKROLL     = 500_000.0
MODEL_UNCERTAINTY    = 0.05
MARGIN_CEIL_RICH     = 0.88
MARGIN_CEIL_MID      = 0.80
MARGIN_CEIL_LEAN     = 0.70
BANKROLL_RICH_THRESH = 0.60
BANKROLL_LEAN_THRESH = 0.30
AGGRESSION_K         = 0.35
AGGRESSION_MIDPOINT  = 5.0
MIN_INCREMENT        = 50.0


class LiveAuctionAgent:

    def __init__(self):
        self.bankroll        = INITIAL_BANKROLL
        self.predicted_value = 0.0
        self.round_number    = 0
        self.cars_processed  = 0

        base_path = os.path.dirname(os.path.abspath(__file__))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model    = joblib.load(os.path.join(base_path, "model_OmSharma.pkl"))
            self.encoders = joblib.load(os.path.join(base_path, "encoders_OmSharma.pkl"))

    def analyze_item(self, item_features: dict):
        self.round_number = 0

        year      = int(item_features.get('year', 2010))
        condition = float(item_features.get('condition', 3.0))
        odometer  = float(item_features.get('odometer', 50_000))
        condition = max(condition, 0.1)

        age_years         = max(REFERENCE_YEAR - year, 1)
        usage_intensity   = odometer / age_years
        odo_per_condition = odometer / condition
        cond_x_year       = condition * year

        features = [
            float(year),
            float(condition),
            float(odometer),
            float(usage_intensity),
            float(age_years),
            float(odo_per_condition),
            float(cond_x_year),
        ]

        for col in CATEGORICAL_COLS:
            raw_val = str(item_features.get(col, ""))
            encoder = self.encoders[col]
            encoded = int(encoder.transform([raw_val])[0]) if raw_val in encoder.classes_ else 0
            features.append(encoded)

        x = np.array([features])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_pred = self.model.predict(x)[0]

        self.predicted_value = float(np.expm1(log_pred))

    def place_bid(self, current_highest_bid: float) -> float:
        self.round_number += 1
        r = self.round_number

        V_adj = self.predicted_value * (1.0 - MODEL_UNCERTAINTY)

        bankroll_ratio = self.bankroll / INITIAL_BANKROLL
        if bankroll_ratio >= BANKROLL_RICH_THRESH:
            base_ceiling = MARGIN_CEIL_RICH
        elif bankroll_ratio >= BANKROLL_LEAN_THRESH:
            t = ((bankroll_ratio - BANKROLL_LEAN_THRESH) /
                 (BANKROLL_RICH_THRESH - BANKROLL_LEAN_THRESH))
            base_ceiling = MARGIN_CEIL_LEAN + t * (MARGIN_CEIL_MID - MARGIN_CEIL_LEAN)
        else:
            base_ceiling = MARGIN_CEIL_LEAN

        sigma    = 1.0 / (1.0 + math.exp(-AGGRESSION_K * (r - AGGRESSION_MIDPOINT)))
        headroom = base_ceiling * 0.10
        max_bid  = V_adj * (base_ceiling + headroom * sigma)
        max_bid  = min(max_bid, V_adj, self.bankroll)

        if current_highest_bid >= max_bid:
            return 0.0

        gap      = max_bid - current_highest_bid
        f_r      = 0.40 + 0.50 * sigma
        next_bid = current_highest_bid + max(gap * f_r, MIN_INCREMENT)
        next_bid = min(next_bid, max_bid, self.bankroll)

        if next_bid <= current_highest_bid:
            return 0.0

        return round(next_bid, 2)

    def auction_result(self, won, winning_bid, actual_price, current_bankroll):
        self.bankroll       = current_bankroll
        self.cars_processed += 1
