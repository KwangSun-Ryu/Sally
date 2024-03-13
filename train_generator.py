import warnings
warnings.simplefilter(action='ignore', category=Warning)

import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer
from ganblr.models import GANBLRPP
from ttgan.synthesizer import TTGANWrapper


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "liver", "mimic", "lung"], default="breast")
parser.add_argument('--model', choices=["CTGAN-O", "CopulaGAN-O", "GANBLRPP-O", "TTGAN-O", "CTGAN-CAT", "CopulaGAN-CAT", "GANBLRPP-CAT", "TTGAN-CAT"], default="CTGAN-O")
args = parser.parse_args()

# Model name and train data
model_name = args.model.split('-')[0]
train_data = args.model.split('-')[1]

# Load data
if train_data == "CAT":
    data = pd.read_csv(f"data/original/{args.name}/discretized/train.csv")
elif train_data == "O":
    data = pd.read_csv(f"data/original/{args.name}/encoded/train.csv")

# Metadata
if model_name in ["CTGAN", "CopulaGAN", "TTGAN"]:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)

# Target
if model_name in ["GANBLRPP", "TTGAN"]:
    columns = json.load(open(f"data/original/{args.name}/columns.json", "r"))
    numerical_columns = []
    for index, column in enumerate(columns):
        if "target" in column:
            target = column["name"]
        if column["type"] == "numerical":
            numerical_columns.append(index)

# Model
if model_name == "CTGAN":
    model = CTGANSynthesizer(
        metadata    = metadata,
        verbose     = True,
        epochs      = 2000,
        checkpoint  = f"checkpoints/{args.name}/{args.model}",
    )
elif model_name == "CopulaGAN":
    model = CopulaGANSynthesizer(
        metadata    = metadata,
        verbose     = True,
        epochs      = 2000,
        checkpoint  = f"checkpoints/{args.name}/{args.model}",
    )
elif model_name == "GANBLRPP":
    model = GANBLRPP(numerical_columns)
elif model_name == "TTGAN":
    model = TTGANWrapper(
        metadata    = metadata,
        target      = target,
        verbose     = True,
        epochs      = 2000,
        checkpoint  = f"checkpoints/{args.name}/{args.model}",
    )

# Train
Path(f"checkpoints/{args.name}/{args.model}").mkdir(parents=True, exist_ok=True)
if model_name in ["CTGAN", "CopulaGAN", "TTGAN"]:
    model.fit(data)
elif model_name == "GANBLRPP":
    X = data.drop(columns=[target]).values
    y = data[target].values
    model.fit(X, y, epochs=100, checkpoint=f"checkpoints/{args.name}/{args.model}")

# Save
Path(f"models/generators/{args.name}").mkdir(parents=True, exist_ok=True)
pickle.dump(model, open(f"models/generators/{args.name}/{args.model}.pkl", "wb"))