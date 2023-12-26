import fnmatch
import os
from typing import List

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from diagram_gen.schemas.exp_file import ExpFile
from diagram_gen.utils.diagram_loader import load_diagram_data
from diagram_gen.utils.file_loader import find_matching_files


def plot_diagram(exp_files: List[ExpFile], tag: str) -> None:
    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == tag:
                    steps, values = zip(*values)
                    plt.plot(steps, values, label=f"{e.path}", alpha=0.7)
        else:
            print(f"Diagram data for {e.path} is None")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"{diagram_type.capitalize()} Diagram - {diagram_name}")
    plt.show()


def aa(data, diagram_type, diagram_name, run_names):
    for tag, values in data.items():
        steps, values = zip(*values)
        for i, run_name in enumerate(run_names):
            if diagram_type == "line":
                plt.plot(steps, values, label=f"{tag} - {run_name}", alpha=0.7)
            elif diagram_type == "boxplot":
                plt.boxplot(
                    values,
                    positions=[step + i * 0.2 for step in steps],
                    labels=[f"{tag} - {run_name}"] * len(steps),
                )
        # Add more diagram types as needed

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"{diagram_type.capitalize()} Diagram - {diagram_name}")
    plt.imshow()
    plt.show()


if __name__ == "__main__":
    # Config variables
    env_name = "../resources/p_my_coin_game"
    experiment_label = "default"
    diagram_name = "eval/efficiency"  # Replace with the actual diagram name
    diagram_type = (
        "line"  # Replace with the desired diagram type ('line', 'boxplot', etc.)
    )

    experiments: List[ExpFile] = find_matching_files(
        exp_path=env_name, exp_label=experiment_label
    )

    for exp in experiments:
        exp.diagram_data = load_diagram_data(path=exp.path, tag=diagram_name)

    plot_diagram(exp_files=experiments, tag=diagram_name)
