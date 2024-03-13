import warnings
warnings.simplefilter(action='ignore', category=Warning)

import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "liver", "mimic", "lung"], default="breast")
parser.add_argument('--model', choices=["RF", "CB", "XGB", "LGBM"], default="RF")
args = parser.parse_args()

# Load data
train_encoded = pd.read_csv(f"data/original/{args.name}/encoded/train.csv")
train_discretized = pd.read_csv(f"data/original/{args.name}/discretized/train.csv")

# Columns
columns = json.load(open(f"data/original/{args.name}/columns.json", "r"))
discretize_columns = []
for column in columns:
    if column["type"] == "numerical":
        discretize_columns.append(column["name"])

# Regression
for column in discretize_columns:
    print("Working on", args.name, args.model, column)
    # Model
    if args.model == "RF":
        model = RandomForestRegressor(random_state=42)
    elif args.model == "CB":
        model = CatBoostRegressor(verbose=0, random_state=42)
    elif args.model == "XGB":
        model = XGBRegressor(random_state=42)
    elif args.model == "LGBM":
        model = LGBMRegressor(verbose=-1, random_state=42)
    # Train
    model.fit(train_discretized, train_encoded[column])
    # Save
    Path(f"models/predictors/{args.name}/{args.model}").mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(f"models/predictors/{args.name}/{args.model}/{column}.pkl", "wb"))