import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "liver", "mimic", "lung"], default="breast")
args = parser.parse_args()

# Load data
train = pd.read_csv(f"data/original/{args.name}/train.csv")
val = pd.read_csv(f"data/original/{args.name}/val.csv")
test = pd.read_csv(f"data/original/{args.name}/test.csv")

# Columns
columns = json.load(open(f"data/original/{args.name}/columns.json", "r"))
encode_columns = []; discretize_columns = []
for column in columns:
    if column["dtype"] in ["object", "bool"]:
        encode_columns.append(column["name"])
    if column["type"] == "numerical":
        discretize_columns.append(column["name"])

# Encode
encoder = OrdinalEncoder()
train[encode_columns] = encoder.fit_transform(train[encode_columns]).astype(int)
val[encode_columns] = encoder.transform(val[encode_columns]).astype(int)
test[encode_columns] = encoder.transform(test[encode_columns]).astype(int)

# Save encoded data
Path(f"data/original/{args.name}/encoded").mkdir(parents=True, exist_ok=True)
train.to_csv(f"data/original/{args.name}/encoded/train.csv", index=False)
val.to_csv(f"data/original/{args.name}/encoded/val.csv", index=False)
test.to_csv(f"data/original/{args.name}/encoded/test.csv", index=False)
pickle.dump(encoder, open(f"data/original/{args.name}/encoded/encoder.pkl", "wb"))

# Discretize
discretizer = KBinsDiscretizer(
    encode      = 'ordinal',
    strategy    = 'kmeans',
)
train[discretize_columns] = discretizer.fit_transform(train[discretize_columns]).astype(int)
val[discretize_columns] = discretizer.transform(val[discretize_columns]).astype(int)
test[discretize_columns] = discretizer.transform(test[discretize_columns]).astype(int)

# Save discretized data
Path(f"data/original/{args.name}/discretized").mkdir(parents=True, exist_ok=True)
train.to_csv(f"data/original/{args.name}/discretized/train.csv", index=False)
val.to_csv(f"data/original/{args.name}/discretized/val.csv", index=False)
test.to_csv(f"data/original/{args.name}/discretized/test.csv", index=False)
pickle.dump(discretizer, open(f"data/original/{args.name}/discretized/discretizer.pkl", "wb"))