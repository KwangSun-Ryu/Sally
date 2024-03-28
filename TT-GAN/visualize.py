import warnings
warnings.simplefilter(action='ignore', category=Warning)

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', choices=["breast", "liver", "mimic", "lung", "diabetes"], default="breast")
parser.add_argument('--validator', choices=["RF", "CB", "XGB", "LGBM"], default="RF")
args = parser.parse_args()


def plots(results, name):
    # Set plot style
    sns.set_style('ticks')
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(12, 4))

    # Draw line plot
    for result in results:
        if glob(f"evaluation/performance/val/{args.name}/{result}.csv"):
            df = pd.read_csv(f"evaluation/performance/val/{args.name}/{result}.csv")
            sns.lineplot(x='Epoch', y='AUC', data=df, lw=0.5, label="-".join(result.split("_")[:-1]) + f" ({df['AUC'].max()}%)")

    # Set plot title and axes labels
    plt.title(f'{args.name.capitalize()} ({args.validator} classifier)')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    
    # Add a legend
    plt.legend(loc='lower right')
    
    # Add a grid
    plt.grid(True)
    
    # Remove the top and right spines
    sns.despine()
    
    # Show the plot
    plt.savefig(f"visualization/{args.name}/{args.validator}/{name}.png", dpi=300, bbox_inches='tight')

Path(f"visualization/{args.name}/{args.validator}").mkdir(parents=True, exist_ok=True)

results = [
    f"CTGAN_O_{args.validator}",
    f"CopulaGAN_O_{args.validator}",
    f"TTGAN_CAT_XGB_{args.validator}",
]
plots(results, "Summary")

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
    f"TTGAN_O_{args.validator}",
    f"TTGAN_CAT_RF_{args.validator}",
    f"TTGAN_CAT_CB_{args.validator}",
    f"TTGAN_CAT_XGB_{args.validator}",
    f"TTGAN_CAT_LGBM_{args.validator}",
]
plots(results, "TTGAN")