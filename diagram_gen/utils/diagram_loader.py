import logging
from typing import Dict, List, Optional, Union

from tensorboard.backend.event_processing import event_accumulator


def load_diagram_data(path: str, tag: Optional[Union[List[str], str]]) -> Dict:
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    diagram_data: Dict = {}
    if tag is None:
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            diagram_data[tag] = [(event.step, event.value) for event in events]
    else:
        if isinstance(tag, str):
            tag = [tag]
        for t in tag:
            if t not in event_acc.Tags()["scalars"]:
                logging.warning(f"Tag {t} not found in {path}")
            else:
                diagram_data[t] = [
                    (event.step, event.value) for event in event_acc.Scalars(t)
                ]
    return diagram_data
