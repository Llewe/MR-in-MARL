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

NUM_COLORS = 20
cm = plt.get_cmap("gist_rainbow")
colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]


def write_scalars(
    axis: Axes,
    e: Union[ExpFile, str],
    values: Union[DataFrame, Any],
    x_label: str,
    y_label: str,
    print_final_value: bool = True,
) -> None:
    df: DataFrame
    if isinstance(values, DataFrame):
        df = values
    else:
        steps, values = zip(*values)
        df = DataFrame({"Steps": steps, "Values": values})

    df["Rolling_Avg"] = df["Values"].rolling(window=10, min_periods=1).mean()

    final_value = df["Rolling_Avg"].iloc[-1]

    line_name: str
    if isinstance(e, str):
        line_name = e
    else:
        line_name = getattr(e.cfg, "NAME", e.path)
    axis.plot(
        df["Steps"],
        df["Rolling_Avg"],
        label=f"{line_name} ({final_value:.2f})"
        if print_final_value
        else f"{line_name}",
        alpha=0.7,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    axis.set_xlim(xmin=0, xmax=max(df["Steps"]))


def get_and_print_scalar_data(
    wanted_data: Dict[str, Tuple[Axes, str, str]],
    exp_files: List[ExpFile],
    merge_same_name: bool = True,
    print_final_value: bool = True,
) -> Dict[str, Dict[str, Union[DataFrame, List[DataFrame]]]]:
    """

    Parameters
    ----------
    print_final_value
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
                write_scalars(ax, line_name, value, x_label, y_label, print_final_value)
            else:
                for d in value:
                    write_scalars(ax, line_name, d, x_label, y_label, print_final_value)

    return data


def save_legend(config: PlotConfig, handles, labels, name: str) -> None:
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    fig = Figure()
    fig.legend(
        handles,
        labels,
        ncols=4,
        loc="center",
        frameon=False,
    )
    config.save(fig, f"{name}-legend")


def plot_ipd(
    exp_files: List[ExpFile],
    output_file: str = "plot",
    merge_same_name: bool = True,
    print_final_value: bool = True,
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)

    fig_eff, ax_eff = plt.subplots()

    wanted_data: Dict[str, Tuple[Axes, str, str]] = {
        "pd-eval/efficiency_per_step": (ax_eff, "Epoche", "Effizienz pro Zug"),
    }

    get_and_print_scalar_data(
        wanted_data=wanted_data,
        exp_files=exp_files,
        merge_same_name=merge_same_name,
        print_final_value=print_final_value,
    )

    handles, labels = ax_eff.get_legend_handles_labels()
    # sort both labels and handles by labels
    save_legend(config, handles, labels, f"{output_file}-efficiency")
    config.save(fig_eff, f"{output_file}-efficiency")

    # tables
    wins_p0: dict[str, float] = {}
    wins_p1: dict[str, float] = {}
    draws: dict[str, float] = {}
    efficiency: dict[str, float] = {}

    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == "pd-eval/wins_p0":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    wins_p0[line_name] = values[-1]
                if t == "pd-eval/wins_p1":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    wins_p1[line_name] = values[-1]
                if t == "pd-eval/draws":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    draws[line_name] = values[-1]
                if t == "pd-eval/efficiency_per_step":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    efficiency[line_name] = values[-1]

        else:
            print(f"Diagram data for {e.path} is None")

    # Plot table (wins)

    fig_wins, ax_wins = plt.subplots()

    #  wins_p0_per_step = (winner_p0 - winner_p1) / history_len

    # Erstelle eine Tabelle mit den Daten
    table_data = []
    for k in wins_p0.keys():
        table_data.append([k, efficiency[k], wins_p0[k], wins_p1[k], draws[k]])
    table = ax_wins.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colLabels=[
            "Algorithmus",
            "Effizienz",
            "Siege Sp. 0",
            "Siege Sp. 1",
            "Unentschieden",
        ],
    )

    # Set column widths based on the maximum string length in each column
    for i, col in enumerate(table_data[0]):
        col_width = max([len(str(row[i])) for row in table_data])
        table.auto_set_column_width([i])

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
    print_final_value=False,
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    fig_eff, ax_eff = plt.subplots()
    fig_own_coin, ax_own_coin = plt.subplots()
    fig_coins_total, ax_coins_total = plt.subplots()

    # ax_eff.set_prop_cycle(color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)])

    wanted_data: Dict[str, Tuple[Axes, str, str]] = {
        "coin_game-eval/coins/own_coin/": (
            ax_own_coin,
            "Epoche",
            "Anteil eigene Münzen",
        ),
        "eval/efficiency": (ax_eff, "Epoche", "Effizienz"),
        "coin_game-eval/coins/total/": (
            ax_coins_total,
            "Epoche",
            "gesammt einsammelte Münzen",
        ),
    }

    get_and_print_scalar_data(
        wanted_data=wanted_data,
        exp_files=exp_files,
        merge_same_name=merge_same_name,
        print_final_value=print_final_value,
    )

    # Efficiency
    handles, labels = ax_eff.get_legend_handles_labels()
    save_legend(config, handles, labels, f"{output_file}-efficiency")
    config.save(fig_eff, f"{output_file}-efficiency")

    # Own Coin
    handles, labels = ax_own_coin.get_legend_handles_labels()
    save_legend(config, handles, labels, f"{output_file}-own_coin")
    config.save(fig_own_coin, f"{output_file}-own_coin")

    # Coins Total
    handles, labels = ax_coins_total.get_legend_handles_labels()
    save_legend(config, handles, labels, f"{output_file}-coins_total")
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
    remove_experiments: List[str] = [
        #   "MATE-Static",
        #   "MATE-Value-Decompose",
        #   "Gifting-Replenishable-Budget",
        #   "Gifting-Fixed-Budget",
        # "RMP-MID (AC, x=0.5)",
        # "RMP-MID (AC, x=1)",
        # "RMP-MID (AC, x=2)",
        "Zentraler Prozentsatz - [0.8]",
        "Zentraler Prozentsatz - [1.0]",
        "Zentrale AC-Strafe - [-0.5-0.5]",
        "Zentrale AC-Strafe - [0-1.5]",
        "Zentrale AC-Strafe - [0-0.5]",
        "Individuelle Metric AC-Strafe - [-0.5-0.5]",
        "Individuelle Metric AC-Strafe - [0-0.5]",
        # "Individuelle Metric AC-Strafe - [0-1.5]",
        "Individuelle AC-Strafe - [-0.5-0.5]",
        "Individuelle AC-Strafe - [0-0.5]",
        # "Individuelle AC-Strafe - [0-1.5]",
        "Gifting-ZS [0.5]",
        "Gifting-ZS [1.5]",
    ]

    replace_dict: Dict[str, str] = {
        "Actor-Critic": "Native Learner",
        "Zentrale AC-Strafe": "RMP Stufe 1",
        "Individuelle Metric AC-Strafe": "RMP Stufe 2",
        "Individuelle AC-Strafe": "RMP Stufe 3",
        "Gifting-ZS [1]": "Gifting-ZS",
    }

    env_name = "../resources/p_coin_game"

    # experiment_label = "4pl-5000"

    # env_name = "../resources/p_prisoners_dilemma"
    experiment_label = "final-5000"

    experiments: List[ExpFile] = find_matching_files(
        exp_path=env_name, exp_label=experiment_label
    )

    for i, e in reversed(list(enumerate(experiments))):
        line_name = getattr(e.cfg, "NAME", e.path)
        if remove_experiments and line_name in remove_experiments and line_name != "":
            experiments.remove(e)

    for e in experiments:
        assert e.cfg is not None
        name = getattr(e.cfg, "NAME", "")
        for k, v in replace_dict.items():
            if k in name:
                e.cfg.NAME = name.replace(k, v)  # .split(" - ")[0]

    for exp in experiments:
        exp.diagram_data = load_diagram_data(path=exp.path, tag=None)
    # draw_heatmaps(experiments, diagram_name)

    # plot_ipd(exp_files=experiments, output_file="ipd", print_final_value=False)

    plots_coin_game(
        exp_files=experiments, output_file="coin_game-short", print_final_value=False
    )
