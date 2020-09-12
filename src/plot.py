import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

RUNS_FOLDER = "runs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs", nargs="*", help="Names of all runs's folders to print"
    )
    parser.add_argument(
        "--savepath",
        type=str,
        help="path to a folder to save training checkpoint and tensorboard file",
    )
    args = parser.parse_args()
    folders = args.runs
    savepath = args.savepath

    results = []
    for folder in folders:
        path_to_folder = Path(RUNS_FOLDER) / folder
        if not path_to_folder.exists():
            raise FileNotFoundError(f'The folder {path_to_folder} does not exist')
        path_to_scores = path_to_folder / "scores.csv"
        path_to_params = path_to_folder / "params.json"
        scores = pd.read_csv(path_to_scores)
        with path_to_params.open("r"):
            params = json.load(path_to_params.open())

        name = params["model"]
        scores = scores.set_index("Objective")
        scores.columns = [f'{col_name} - {name}' for col_name in scores.columns]
        results.append(scores)

    comp = results[0] if len(results) == 1 else results[0].join(results[1:])
    
    kl_results = comp[comp.columns[comp.columns.str.contains("KL")]]
    js_results = comp[comp.columns[comp.columns.str.contains("JS")]]

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    kl_results.plot(ax=axes[0])
    js_results.plot(ax=axes[1])
    if savepath:
        plt.savefig(savepath)    
    plt.show()
