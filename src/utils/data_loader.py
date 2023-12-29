import json
from enum import Enum

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


# Example usage:
if __name__ == "__main__":
    # Define a Pydantic model
    class Person(BaseModel):
        class A(str, Enum):
            a = "a"
            b = "b"
            c = "c"

        name: str
        age: int
        a: A

    # Load data from the file (if it exists)
    loaded_data = load_pydantic_object("person.json", Person)

    if loaded_data:
        print("Loaded Data:")
        print(loaded_data)
    else:
        # If the file doesn't exist, create a new instance
        new_person = Person(name="John Doe", age=30, a=Person.A.b)
        print("New Data:")
        print(new_person)

        # Save the new data to the file
        save_pydantic_object("person.json", new_person)
        print("Data saved to 'person.yml'")
