import json

import yaml
from pydantic import BaseModel


def load_pydantic_object(file_path: str, model_class):
    try:
        with open(file_path, "r") as file:
            if file_path.endswith(".json"):
                data = json.load(file)
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                data = yaml.safe_load(file)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")
        return model_class(**data)
    except FileNotFoundError:
        return None


def save_pydantic_object(file_path: str, data: BaseModel):
    with open(file_path, "w") as file:
        if file_path.endswith(".json"):
            json.dump(data.model_dump(), file, indent=4)
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            yaml.dump(data.model_dump(), file, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml")
