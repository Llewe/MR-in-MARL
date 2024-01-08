from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
from pylab import Figure, Axes
from tensorboard.backend.event_processing import event_accumulator

from diagram_gen.config.plot_config import PlotConfig
from diagram_gen.schemas.exp_file import ExpFile
from diagram_gen.utils.diagram_loader import load_diagram_data
from diagram_gen.utils.file_loader import find_matching_files


def write_scalars(
    axis: Axes,
    e: Union[ExpFile, str],
    values: Union[DataFrame, Any],
    x_label: str,
    y_label: str,
) -> None:
    df: DataFrame
    if isinstance(values, DataFrame):
        df = values
    else:
        steps, values = zip(*values)
        df = DataFrame({"Steps": steps, "Values": values})

    df["Rolling_Avg"] = df["Values"].rolling(window=3, min_periods=1).mean()

    line_name: str
    if isinstance(e, str):
        line_name = e
    else:
        line_name = getattr(e.cfg, "NAME", e.path)

    axis.plot(
        df["Steps"],
        df["Rolling_Avg"],
        label=f"{line_name}",
        alpha=0.7,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    axis.set_xlim(xmin=0, xmax=max(df["Steps"]))


def get_and_print_scalar_data(
    wanted_data: Dict[str, Tuple[Axes, str, str]],
    exp_files: List[ExpFile],
    merge_same_name: bool = True,
) -> Dict[str, Dict[str, Union[DataFrame, List[DataFrame]]]]:
    """

    Parameters
    ----------
    wanted_data: Dict[str, Tuple[Axes, str, str]]
        Dict with the wanted data. The key is the tag and the value is a tuple with the axis, x_label and y_label

    exp_files: List[ExpFile]
        List of experiments

    merge_same_name: bool
        Merge experiments with the same name (default: True)

    Returns
    -------

    """
    tags = [k for k in wanted_data.keys()]
    data: Dict[str, Dict[str, Union[DataFrame, List[DataFrame]]]] = {}

    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t in tags:
                    if t not in data:
                        data[t] = {}
                    line_name = getattr(e.cfg, "NAME", e.path)
                    if line_name == "":
                        line_name = e.path
                    if t not in data:
                        data[t] = {}
                    if line_name not in data[t]:
                        data[t][line_name] = []

                    steps, values = zip(*values)
                    df = pd.DataFrame({"Steps": steps, "Values": values})
                    data[t][line_name].append(df)

        else:
            print(f"Diagram data for {e.path} is None")

    # Merging and calculating the average only when merge is True
    if merge_same_name:
        for tag, lines in data.items():
            for line_name, df_list in lines.items():
                if len(df_list) > 1:
                    merged_df = pd.concat(df_list, ignore_index=True)
                    avg_df = merged_df.groupby("Steps")["Values"].mean().reset_index()
                    data[tag][line_name] = avg_df
                else:
                    data[tag][line_name] = df_list[0]

    for tag, (ax, x_label, y_label) in wanted_data.items():
        for line_name, value in data[tag].items():
            if merge_same_name:
                write_scalars(ax, line_name, value, x_label, y_label)
            else:
                for d in value:
                    write_scalars(ax, line_name, d, x_label, y_label)

    return data


def plot_ipd(
    exp_files: List[ExpFile],
    output_file: str = "plot",
    merge_same_name: bool = True,
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)

    fig_eff, ax_eff = plt.subplots()

    wanted_data: Dict[str, Tuple[Axes, str, str]] = {
        "pd-eval/efficiency_per_step": (ax_eff, "Epoch", "Efficiency pro Step")
    }

    get_and_print_scalar_data(
        wanted_data=wanted_data,
        exp_files=exp_files,
        merge_same_name=merge_same_name,
    )

    fig_eff.legend(ncols=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.1))
    config.save(fig_eff, f"{output_file}-efficiency")

    # tables
    wins: dict[str, float] = {}

    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == "pd-eval/wins_p0_per_step":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    wins[line_name] = values[-1]

        else:
            print(f"Diagram data for {e.path} is None")

    # Plot table (wins)

    fig_wins, ax_wins = plt.subplots()

    #  wins_p0_per_step = (winner_p0 - winner_p1) / history_len

    # Erstelle eine Tabelle mit den Daten
    table_data = []
    for key, values in wins.items():
        table_data.append([key, values])

    table = ax_wins.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colLabels=["Algorithm", "Siege Spieler 0"],
    )

    # Konfiguriere das Aussehen der Tabelle
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Verstecke die Achsen
    ax_wins.axis("off")
    config.save(fig_wins, f"{output_file}-wins")


def plots_coin_game(
    exp_files: List[ExpFile],
    output_file: str = "plot",
    merge_same_name: bool = True,
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    fig_eff, ax_eff = plt.subplots()
    fig_own_coin, ax_own_coin = plt.subplots()
    fig_coins_total, ax_coins_total = plt.subplots()

    wanted_data: Dict[str, Tuple[Axes, str, str]] = {
        "coin_game-eval/coins/own_coin/": (ax_own_coin, "Epoch", "Own Coin"),
        "eval/efficiency": (ax_eff, "Epoch", "Efficiency"),
        "coin_game-eval/coins/total/": (ax_coins_total, "Epoch", "Total Coins"),
    }

    get_and_print_scalar_data(
        wanted_data=wanted_data,
        exp_files=exp_files,
        merge_same_name=merge_same_name,
    )

    fig_eff.legend(ncols=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.15))
    config.save(fig_eff, f"{output_file}-efficiency")
    fig_own_coin.legend(
        ncols=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.15)
    )
    config.save(fig_own_coin, f"{output_file}-own_coin")
    fig_coins_total.legend(
        ncols=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.15)
    )
    config.save(fig_coins_total, f"{output_file}-coins_total")


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

    env_name = "../resources/p_coin_game"
    experiment_label = "n-4pl-5000"

    # env_name = "../resources/p_prisoners_dilemma"
    # experiment_label = "default-5000 (copy)"

    experiments: List[ExpFile] = find_matching_files(
        exp_path=env_name, exp_label=experiment_label
    )

    for exp in experiments:
        exp.diagram_data = load_diagram_data(path=exp.path, tag=None)
    # draw_heatmaps(experiments, diagram_name)
    # plot_ipd(exp_files=experiments, output_file="ipd")

    plots_coin_game(exp_files=experiments, output_file="coin_game")
