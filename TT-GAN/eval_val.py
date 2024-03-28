import warnings
warnings.simplefilter(action='ignore', category=Warning)

import torch
import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from alive_progress import alive_bar


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "liver", "mimic", "lung"], default="breast")
parser.add_argument('--model', choices=["CTGAN-O", "CopulaGAN-O", "TTGAN-O", "CTGAN-CAT", "CopulaGAN-CAT", "TTGAN-CAT"], default="CTGAN-O")
parser.add_argument('--predictor', choices=["RF", "CB", "XGB", "LGBM"], default="RF")
parser.add_argument('--validator', choices=["RF", "CB", "XGB", "LGBM"], default="RF")
args = parser.parse_args()

# Model name and train data
model_name = args.model.split('-')[0]
train_data = args.model.split('-')[1]
if train_data == "O":
    filename = f"{model_name}_{train_data}_{args.validator}"
else:
    filename = f"{model_name}_{train_data}_{args.predictor}_{args.validator}"
print(args)

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

# Number of epochs
epochs = 2000

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

# Validation data
val_data = pd.read_csv(f"data/original/{args.name}/val.csv")
val_data[encode_columns] = encoder.transform(val_data[encode_columns]).astype(int)
X_val, y_val = val_data.drop(columns=[target]), val_data[target]

# Create directory
Path(f"evaluation/performance/val/{args.name}").mkdir(parents=True, exist_ok=True)

# Header
with open(f"evaluation/performance/val/{args.name}/{filename}.csv", "w") as f:
    f.write("Epoch,AUC\n")

# Loop
with alive_bar(epochs) as bar:
    for epoch in range(1, epochs + 1):

        # Load model
        model = pickle.load(open(f"models/generators/{args.name}/{args.model}.pkl", "rb"))
        if model_name in ["CTGAN", "CopulaGAN"]:
            model._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{args.model}/checkpoint_{epoch}.pt"))
        elif model_name == "TTGAN":
            model.model_A._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{args.model}/checkpoint_A_{epoch}.pt"))
            model.model_B._model._generator.load_state_dict(torch.load(f"checkpoints/{args.name}/{args.model}/checkpoint_B_{epoch}.pt"))

        # Sample
        while True:
            samples = model.sample(num_samples)
            if len(samples[target].unique()) > 1:
                break

        # # Column type
        # for column in columns:
        #     if column["dtype"] in ["int", "object", "bool"]:
        #         samples[column["name"]] = samples[column["name"]].astype(int)
        #     elif column["dtype"] == "float":
        #         samples[column["name"]] = samples[column["name"]].astype(float).round(column["decimal"])
        # samples[target] = samples[target].astype(int)

        # # Predict numerical columns
        # if train_data == "CAT":
        #     # Predict numerical columns
        #     predicted = {}
        #     for column in discretize_columns:
        #         # Predictor
        #         predictor = pickle.load(open(f"models/predictors/{args.name}/{args.predictor}/{column}.pkl", "rb"))
        #         predicted[column] = predictor.predict(samples)
        #     # Replace numerical columns
        #     for column in discretize_columns:
        #         samples[column] = predicted[column]

        # # Train data
        # X_train, y_train = samples.drop(columns=[target]), samples[target]

        # # Model
        # if args.validator == "RF":
        #     validator = RandomForestClassifier(random_state=42)
        # elif args.validator == "CB":
        #     validator = CatBoostClassifier(verbose=0, random_state=42)
        # elif args.validator == "XGB":
        #     validator = XGBClassifier(random_state=42)
        # elif args.validator == "LGBM":
        #     validator = LGBMClassifier(verbose=-1, random_state=42)

        # # Train
        # validator.fit(X_train, y_train)

        # # Predict
        # y_proba = validator.predict_proba(X_val)[:, 1]

        # # Save
        # with open(f"evaluation/performance/val/{args.name}/{filename}.csv", "a") as f:
        #     f.write(f"{epoch:.0f},{roc_auc_score(y_val, y_proba) * 100:.2f}\n")

        # Update bar
        bar()

    from collections import Counter
    print(Counter(samples[target]))