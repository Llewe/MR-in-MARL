from io import BytesIO
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
from pylab import Figure
from tensorboard.backend.event_processing import event_accumulator

from diagram_gen.config.plot_config import PlotConfig
from diagram_gen.schemas.exp_file import ExpFile
from diagram_gen.utils.diagram_loader import load_diagram_data
from diagram_gen.utils.file_loader import find_matching_files


def write_scalars(
    axis, fig: Optional[Figure], e: ExpFile, values, x_label: str, y_label: str
) -> None:
    steps, values = zip(*values)

    df = DataFrame({"Steps": steps, "Values": values})
    df["Rolling_Avg"] = df["Values"].rolling(window=3, min_periods=1).mean()

    line_name = getattr(e.cfg, "NAME", e.path)
    axis.plot(
        df["Steps"],
        df["Rolling_Avg"],
        label=f"{line_name}",
        alpha=0.7,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    axis.set_xlim(xmin=0)
    if fig is not None:
        fig.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.15))


def plot_ipd(
    exp_files: List[ExpFile],
    output_file: str = "plot",
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    fig_eff, ax_eff = plt.subplots()
    wins: dict[str, float] = {}

    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == "pd-eval/efficiency_per_step":
                    write_scalars(
                        ax_eff, fig_eff, e, values, "Epoch", "Efficiency pro Step"
                    )
                if t == "pd-eval/wins_p0_per_step":
                    steps, values = zip(*values)

                    line_name = getattr(e.cfg, "NAME", e.path)

                    wins[line_name] = values[-1]

        else:
            print(f"Diagram data for {e.path} is None")

    config.save(fig_eff, f"{output_file}-efficiency")

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
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    fig_eff, ax_eff = plt.subplots()
    fig_own_coin, ax_own_coin = plt.subplots()
    wins: dict[str, float] = {}

    for e in exp_files:
        if e.diagram_data is not None:
            for t, values in e.diagram_data.items():
                if t == "eval/efficiency":
                    write_scalars(ax_eff, None, e, values, "Epoch", "Efficiency")
                if t == "coin_game-eval/coins/own_coin/":
                    write_scalars(ax_own_coin, None, e, values, "Epoch", "Own Coin")

        else:
            print(f"Diagram data for {e.path} is None")

    fig_eff.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.25))
    config.save(fig_eff, f"{output_file}-efficiency")
    fig_own_coin.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.25))
    config.save(fig_own_coin, f"{output_file}-own_coin")


def plot_diagram(
    exp_files: List[ExpFile],
    tag: str,
    x_label: str = "Epoch",
    y_label: str = "Efficiency pro Step",
    output_file: str = "plot",
) -> None:
    config: PlotConfig = PlotConfig()

    config.configPlt(plt)
    # graph
    fig, ax = plt.subplots()

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

                    line_name = getattr(e.cfg, "NAME", e.path)
                    ax.plot(
                        df["Steps"],
                        df["Rolling_Avg"],
                        label=f"{line_name}",
                        alpha=0.7,
                    )

        else:
            print(f"Diagram data for {e.path} is None")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.legend(
        loc="outside left upper",
        ncols=2,
    )
    # plt.title(f"{diagram_type.capitalize()} Diagram - {diagram_name}")
    #
    # # table
    #
    # table = []
    # for e in exp_files:
    #     if e.cfg is not None:
    #         dict = e.cfg.model_dump()
    #         dict["type"] = e.agent_type
    #         dict["path"] = e.path.split("/")[-2].split(" - ")[-1]
    #         table.append(dict)
    #
    # dfTable = DataFrame(table)
    #
    # cell_text = []
    # for row in range(len(dfTable)):
    #     cell_text.append(dfTable.iloc[row])
    #
    # plt.table(cellText=cell_text, colLabels=dfTable.columns, loc="top")

    config.save(fig, output_file)


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
    diagram_type = (
        "line"  # Replace with the desired diagram type ('line', 'boxplot', etc.)
    )

    experiments: List[ExpFile] = find_matching_files(
        exp_path=env_name, exp_label=experiment_label
    )

    for exp in experiments:
        exp.diagram_data = load_diagram_data(path=exp.path, tag=None)
    # draw_heatmaps(experiments, diagram_name)
    # plot_diagram(exp_files=experiments, tag=diagram_name, output_file="efficency-ipd")
    # plot_ipd(exp_files=experiments, output_file="ipd")
    plots_coin_game(exp_files=experiments, output_file="coin_game")
