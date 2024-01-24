from statistics import mean
from typing import Dict, List

from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.controller_ma.central_ma_heuristic_coin_game import CentralMaHeuristicCoinGame
from src.enums import EnvType


class HeuristicCompareLogger:
    writer: SummaryWriter
    heuristic_diff: List[float]
    correct_changes: List[float]
    incorrect_changes: List[float]
    no_changes: List[float]
    coin_owner_highest_reward: List[float]
    env_type: EnvType

    sw_orig: float
    sw_manipulated: float

    def __init__(self, writer: SummaryWriter, env_type: EnvType):
        self.writer = writer
        self.heuristic_diff = []
        self.correct_changes = []
        self.incorrect_changes = []
        self.no_changes = []
        self.coin_owner_highest_reward = []

        self.env_type = EnvType(env_type)

    def reset(self):
        self.heuristic_diff.clear()
        self.correct_changes.clear()
        self.incorrect_changes.clear()
        self.no_changes.clear()
        self.coin_owner_highest_reward.clear()

        self.sw_orig = 0
        self.sw_manipulated = 0

    def add(
        self,
        obs: ObsType,
        rewards: Dict[AgentID, float],
        manipulated_rewards: Dict[AgentID, float],
    ):
        sw_orig = abs(sum(rewards.values()))
        sw_manipulated = abs(sum(manipulated_rewards.values()))

        self.sw_orig += sw_orig
        self.sw_manipulated += sw_manipulated

        diff: float = 0
        if self.env_type == EnvType.P_COIN_GAME:
            heuristic = CentralMaHeuristicCoinGame.get_modified_reward(obs, rewards)

            # to do check if correct got punnished

            coin_owner = obs[-1]

            for a in heuristic:
                heuristic_reward = heuristic[a]
                real_reward = rewards[a]
                manipulated_reward = manipulated_rewards[a]

                diff += abs(heuristic_reward - manipulated_reward)

                if heuristic_reward > real_reward:
                    if manipulated_reward > real_reward:
                        self.correct_changes.append(1)
                    else:
                        self.incorrect_changes.append(1)
                elif heuristic_reward < real_reward:
                    if manipulated_reward < real_reward:
                        self.correct_changes.append(1)
                    else:
                        self.incorrect_changes.append(1)
                elif heuristic_reward == real_reward:
                    self.correct_changes.append(1)
                else:
                    if manipulated_reward == manipulated_reward:
                        self.no_changes.append(1)
                    else:
                        self.incorrect_changes.append(1)

            if (
                max(manipulated_rewards.values())
                == list(manipulated_rewards.values())[coin_owner[0]]
            ):
                self.coin_owner_highest_reward.append(1)
            else:
                self.coin_owner_highest_reward.append(0)
        else:
            raise NotImplementedError
        self.heuristic_diff.append(diff)

    def log_and_clear(self, epoch: int, tag: str):
        self.writer.add_scalar(
            f"{tag}/heuristic_diff", mean(self.heuristic_diff), global_step=epoch
        )

        correct_changes = sum(self.correct_changes)
        incorrect_changes = sum(self.incorrect_changes)
        no_changes = sum(self.no_changes)
        total_changes = correct_changes + incorrect_changes + no_changes

        percentage_correct_changes = correct_changes / total_changes
        percentage_incorrect_changes = incorrect_changes / total_changes
        percentage_no_changes = no_changes / total_changes

        percentage_correct = (correct_changes + no_changes) / total_changes

        coin_owner_highest_reward = sum(self.coin_owner_highest_reward) / len(
            self.coin_owner_highest_reward
        )

        self.writer.add_scalar(
            f"{tag}/sw_diff", self.sw_orig - self.sw_manipulated, global_step=epoch
        )

        self.writer.add_scalar(
            f"{tag}/coin_owner_highest_reward",
            coin_owner_highest_reward,
            global_step=epoch,
        )

        self.writer.add_scalar(
            f"{tag}/percentage_correct_changes",
            percentage_correct_changes,
            global_step=epoch,
        )
        self.writer.add_scalar(
            f"{tag}/percentage_incorrect_changes",
            percentage_incorrect_changes,
            global_step=epoch,
        )
        self.writer.add_scalar(
            f"{tag}/percentage_no_changes", percentage_no_changes, global_step=epoch
        )
        self.writer.add_scalar(
            f"{tag}/percentage_overall_correct", percentage_correct, global_step=epoch
        )

        self.writer.add_scalar(
            f"{tag}/correct_changes", correct_changes, global_step=epoch
        )
        self.writer.add_scalar(
            f"{tag}/incorrect_changes", incorrect_changes, global_step=epoch
        )
        self.writer.add_scalar(f"{tag}/no_changes", no_changes, global_step=epoch)

        self.reset()
