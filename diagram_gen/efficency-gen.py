import fnmatch
import os
from typing import List

import numpy
import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt
from src.training import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from scipy.interpolate import make_interp_spline, BSpline
from diagram_gen.config.plot_config import PlotConfig
from diagram_gen.schemas.exp_file import ExpFile
from diagram_gen.utils.diagram_loader import load_diagram_data
from diagram_gen.utils.file_loader import find_matching_files
from PIL import Image
from matplotlib.ticker import PercentFormatter
from io import BytesIO


def plot_diagram(
    exp_files: List[ExpFile],
    tag: str,
    x_label: str = "Steps",
    y_label: str = "Efficiency",
    output_file: str = "plot",
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == tag:
                    steps, values = zip(*values)

                    # new_steps = np.linspace(min(steps), max(steps), num=len(steps) * 10)
                    # spl: BSpline = make_interp_spline(steps, values, k=7)

                    # smoothed_values = spl(new_steps)

                    # Use numpy.convolve for moving average smoothing
                    df = DataFrame({"Steps": steps, "Values": values})
                    df["Rolling_Avg"] = df["Values"].rolling(window=3).mean()
                    # plt.plot(new_steps, smoothed_values, label=f"{e.path}", alpha=0.7)
                    plt.plot(
                        df["Steps"],
                        df["Rolling_Avg"],
                        label=f"{e.path}",
                        alpha=0.7,
                    )

        else:
            print(f"Diagram data for {e.path} is None")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend()
    plt.title(f"{diagram_type.capitalize()} Diagram - {diagram_name}")

    # table

    table = []
    for e in exp_files:
        if e.cfg is not None:
            dict = e.cfg.model_dump()
            dict["type"] = e.agent_type
            dict["path"] = e.path.split("/")[-2].split(" - ")[-1]
            table.append(dict)

    dfTable = DataFrame(table)

    cell_text = []
    for row in range(len(dfTable)):
        cell_text.append(dfTable.iloc[row])

    plt.table(cellText=cell_text, colLabels=dfTable.columns, loc="top")

    config.save(plt, output_file)
    plt.show()


def draw_heatmaps(
    exp_files: List[ExpFile],
    tag: str,
    x_label: str = "Steps",
    y_label: str = "Efficiency",
    output_file: str = "plot",
) -> None:
    config: PlotConfig = PlotConfig()

    for e in exp_files:
        event_acc = event_accumulator.EventAccumulator(e.path)
        event_acc.Reload()
        for tag in event_acc.Tags()["images"]:
            print(tag)
            events = event_acc.Images(tag)
            print(events)
            imagee: event_accumulator.ImageEvent = events[-1]  # get the last image

            image = Image.open(BytesIO(imagee.encoded_image_string))
            image_array = np.array(image)
            # Create a Matplotlib figure and axis
            fig, ax = plt.subplots()

            # Display the image as a heatmap
            cax = ax.imshow(image_array, cmap="gray", vmin=0, vmax=100)
            # Add colorbar

            cbar = fig.colorbar(cax, format=PercentFormatter())

            # Set title and show the plot
            # ax.set_title("Positions-Heatmap")
            config.save(fig, f"{output_file}-{tag.split('/')[-1]}-step-{imagee.step}")


if __name__ == "__main__":
    # Config variables
    env_name = "../resources/p_my_coin_game"
    experiment_label = "test"
    diagram_name = "eval/efficiency"  # Replace with the actual diagram name
    diagram_type = (
        "line"  # Replace with the desired diagram type ('line', 'boxplot', etc.)
    )

    experiments: List[ExpFile] = find_matching_files(
        exp_path=env_name, exp_label=experiment_label
    )

    for exp in experiments:
        exp.diagram_data = load_diagram_data(path=exp.path, tag=None)
    draw_heatmaps(experiments, diagram_name)
    # plot_diagram(
    #     exp_files=experiments, tag=diagram_name, output_file="efficency-coingame"
    # )