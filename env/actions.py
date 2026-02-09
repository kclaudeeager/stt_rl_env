from enum import Enum


class Action(Enum):
    LR_UP = "lr_up"
    LR_DOWN = "lr_down"
    BATCH_UP = "batch_up"
    BATCH_DOWN = "batch_down"
    FREEZE_MORE = "freeze_more"
    FREEZE_LESS = "freeze_less"
    STOP = "stop"


def is_valid_action(action: Action) -> bool:
    return action in Action
