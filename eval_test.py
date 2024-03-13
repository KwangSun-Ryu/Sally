import warnings
warnings.simplefilter(action='ignore', category=Warning)

import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "lung", "liver", "diabetes", "mimic"], default="breast")
parser.add_argument('--validator', choices=["RF", "CB", "XGB", "LGBM"], default="RF")
args = parser.parse_args()

# Number of samples
if args.name == "breast":
    num_samples = 224
elif args.name == "lung":
    num_samples = 1616
elif args.name == "liver":
    num_samples = 4767
elif args.name == "diabetes":
    num_samples = 71234
elif args.name == "mimic":
    num_samples = 309006

# Columns
columns = json.load(open(f"data/original/{args.name}/columns.json", "r"))
column_names = []; encode_columns = []; discretize_columns = []
for column in columns:
    column_names.append(column["name"])
    if column["dtype"] in ["object", "bool"]:
        encode_columns.append(column["name"])
    if column["type"] == "numerical":
        discretize_columns.append(column["name"])
    if "target" in column:
        target = column["name"]

# Encoder
encoder = pickle.load(open(f"data/original/{args.name}/encoded/encoder.pkl", "rb"))

# Test data
test_data = pd.read_csv(f"data/original/{args.name}/test.csv")
test_data[encode_columns] = encoder.transform(test_data[encode_columns]).astype(int)
X_test, y_test = test_data.drop(columns=[target]), test_data[target]

# Create directory
Path(f"evaluation/performance/test/{args.name}").mkdir(parents=True, exist_ok=True)

# Header
with open(f"evaluation/performance/test/{args.name}/{args.validator}.csv", "w") as f:
    f.write("Method,AUC\n")

def plots(results, name):
    # Draw line plot
    for result in results:
        auc_list = []
        for i in range(10):
            if glob(f"evaluation/performance/val/{args.name}/{result}.csv"):
                df = pd.read_csv(f"evaluation/performance/val/{args.name}/{result}.csv")
                # pandas select row with max AUC
                best_row = df.loc[df['AUC'].idxmax()]
                epoch = int(best_row['Epoch'])
                model_name = "-".join(result.split("_")[:2])
                train_data = result.split("_")[1]
                p = result.split("_")[2]
                
                # Load model
                model = pickle.load(open(f"models/generators/{args.name}/{model_name}.pkl", "rb"))
                if name in ["CTGAN", "CopulaGAN"]:
                    model._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{model_name}/checkpoint_{epoch}.pt"))
                elif name == "TTGAN":
                    model.model_A._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{model_name}/checkpoint_A_{epoch}.pt"))
                    model.model_B._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{model_name}/checkpoint_B_{epoch}.pt"))
                elif name == "GANBLRPP":
                    model.__ganblr = pickle.load(open(f"checkpoints/{args.name}/{model_name}/checkpoint_{epoch}.pkl", "rb"))

                # Sample
                samples = model.sample(num_samples)

                # GANBLRPP
                if name == "GANBLRPP":
                    samples = pd.DataFrame(samples, columns=column_names)
                
                # Column type
                for column in columns:
                    if column["dtype"] in ["int", "object", "bool"]:
                        samples[column["name"]] = samples[column["name"]].astype(int)
                    elif column["dtype"] == "float":
                        samples[column["name"]] = samples[column["name"]].astype(float).round(column["decimal"])
                samples[target] = samples[target].astype(int)

                # Predict numerical columns
                if train_data == "CAT":
                    # Predict numerical columns
                    predicted = {}
                    for column in discretize_columns:
                        # Predictor
                        predictor = pickle.load(open(f"models/predictors/{args.name}/{p}/{column}.pkl", "rb"))
                        predicted[column] = predictor.predict(samples)
                    # Replace numerical columns
                    for column in discretize_columns:
                        samples[column] = predicted[column]

                # Train data
                X_train, y_train = samples.drop(columns=[target]), samples[target]

                # Model
                if args.validator == "RF":
                    validator = RandomForestClassifier(random_state=42)
                elif args.validator == "CB":
                    validator = CatBoostClassifier(verbose=0, random_state=42)
                elif args.validator == "XGB":
                    validator = XGBClassifier(random_state=42)
                elif args.validator == "LGBM":
                    validator = LGBMClassifier(verbose=-1, random_state=42)

                # Train
                validator.fit(X_train, y_train)

                # Predict
                pred = validator.predict_proba(X_test)
                if pred.shape[1] < 2:
                    pred = np.hstack([pred, 1 - pred])  
                y_proba = pred[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                auc_list.append(auc)
        print(f"{result},{np.mean(auc_list) * 100:.2f}")

        # Save
        with open(f"evaluation/performance/test/{args.name}/{args.validator}.csv", "a") as f:
            f.write(f"{result},{np.mean(auc_list) * 100:.2f}\n")


results = [
    f"CTGAN_O_{args.validator}",
    f"CTGAN_CAT_RF_{args.validator}",
    f"CTGAN_CAT_CB_{args.validator}",
    f"CTGAN_CAT_XGB_{args.validator}",
    f"CTGAN_CAT_LGBM_{args.validator}",
]
plots(results, "CTGAN")

results = [
    f"CopulaGAN_O_{args.validator}",
    f"CopulaGAN_CAT_RF_{args.validator}",
    f"CopulaGAN_CAT_CB_{args.validator}",
    f"CopulaGAN_CAT_XGB_{args.validator}",
    f"CopulaGAN_CAT_LGBM_{args.validator}",
]
plots(results, "CopulaGAN")

results = [
    f"GANBLRPP_O_{args.validator}",
    f"GANBLRPP_CAT_RF_{args.validator}",
    f"GANBLRPP_CAT_CB_{args.validator}",
    f"GANBLRPP_CAT_XGB_{args.validator}",
    f"GANBLRPP_CAT_LGBM_{args.validator}",
]
plots(results, "GANBLRPP")

results = [
    f"TTGAN_O_{args.validator}",
    f"TTGAN_CAT_RF_{args.validator}",
    f"TTGAN_CAT_CB_{args.validator}",
    f"TTGAN_CAT_XGB_{args.validator}",
    f"TTGAN_CAT_LGBM_{args.validator}",
]
plots(results, "TTGAN")