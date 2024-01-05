import fnmatch
import os
from typing import Dict, List

from diagram_gen.schemas.exp_file import ExpFile
from src.controller.utils.agents_helper import get_agent_class
from src.config.ctrl_config import CtrlConfig
from src.enums import AgentType
from src.utils.data_loader import load_pydantic_object
from pydantic import BaseModel


def find_matching_files(
    exp_path: str,
    exp_label: str,
    pattern: str = "events.out.tfevents*",
) -> List[ExpFile]:
    exp_dir: str = os.path.join(exp_path, exp_label)
    if not os.path.isdir(exp_dir):
        raise Exception(f"Experiment directory {exp_dir} does not exist")
    matching_files: List[ExpFile] = []

    for root, _, files in os.walk(exp_dir):
        for filename in fnmatch.filter(files, pattern):
            path = os.path.join(root, filename)
            parts = path.split("/")

            controller_file = os.path.join(root, "controller.json")
            configClass = get_agent_class(AgentType(parts[-3]))[1]
            model: CtrlConfig = load_pydantic_object(controller_file, configClass)

            matching_files.append(
                ExpFile(
                    path=path,
                    agent_type=parts[-3],
                    diagram_data=None,
                    cfg=model,
                )
            )

    return matching_files


if __name__ == "__main__":
    # Config variables
    env_name = "../../resources/p_my_coin_game"
    experiment_label = "default"
    diagram_name = "eval/efficiency"  # Replace with the actual diagram name
    diagram_type = (
        "line"  # Replace with the desired diagram type ('line', 'boxplot', etc.)
    )

    # Load and plot all runs for the specified environment and experiment
    for i in find_matching_files(
        env_name,
        experiment_label,
        pattern="events.out.tfevents.1703089875.lanaya.42760.0",
    ):
        print(i.agent_type)
