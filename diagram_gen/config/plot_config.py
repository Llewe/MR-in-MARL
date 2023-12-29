import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class PlotConfig(BaseSettings):
    class Style(str, Enum):
        DEFAULT = None
        PACOTY = "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle"
        PITAYASMOOTHIE = "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle"
        CYBERPUNK = "https://github.com/dhaitz/mplcyberpunk/blob/master/mplcyberpunk/data/cyberpunk.mplstyle"

    fig_width: int = 64
    fig_height: int = 16

    style: Style = Style.DEFAULT

    storage_path: str = "../output"

    def configPlt(self, plot) -> None:
        plot.figure(figsize=(self.fig_width, self.fig_height))

        if self.style is not PlotConfig.Style.DEFAULT:
            plot.style.use(self.style.value)

    def save(self, plot, name: str) -> None:
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        plot.savefig(f"{self.storage_path}/{name}.png", bbox_inches="tight")
