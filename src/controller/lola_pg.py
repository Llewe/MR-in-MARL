from typing import List

import numpy
import torch
from gymnasium.spaces import Space
from pettingzoo.utils.env import AgentID
from src.controller.actor_critic import ActorCritic
from torch import Tensor
from torch.distributions import Categorical
from torch.autograd import grad

from src.config.ctrl_config import LolaPGConfig


class LolaPG(ActorCritic):
    """
    LolaPG (Lola Policy Gradient)
    """

    second_order_lr: float

    gradients: dict[AgentID, [torch.Tensor]]

    def __init__(self, config: LolaPGConfig):
        super(LolaPG, self).__init__(config)

        self.second_order_lr = config.SECOND_ORDER_LR

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super(LolaPG, self).init_agents(action_space, observation_space)

        self.all_gradients = {}

    def preprocess(self) -> None:
        self.all_gradients.clear()

        for agent_i in self.actor_networks:
            print("preprocess agent_i", agent_i)
            actor_i = self.actor_networks[agent_i]

            history_i: A2C.RolloutBuffer = self.step_info[agent_i]

            obs_i = torch.stack(history_i.observations)

            probs_i = actor_i(obs_i.detach())

            reward_i = torch.tensor(
                self.compute_returns(history_i.rewards, self.config.DISCOUNT_FACTOR),
                dtype=torch.float32,
            ).detach()

            reward_i -= torch.stack(history_i.values).squeeze().detach()

            gradients = [torch.zeros_like(weights) for weights in actor_i.parameters()]

            action_i = torch.tensor(history_i.actions, dtype=torch.int64).detach()  # OK

            for agent_j in self.actor_networks:
                print("loop ", agent_j)
                if agent_j != agent_i:
                    actor_j = self.actor_networks[agent_j]

                    history_j: A2C.RolloutBuffer = self.step_info[agent_j]

                    obs_j = torch.stack(history_j.observations).detach()

                    probs_j = actor_j(obs_j.detach())

                    reward_j = torch.tensor(
                        self.compute_returns(
                            history_j.rewards, self.config.DISCOUNT_FACTOR
                        ),
                        dtype=torch.float32,
                    ).detach()

                    reward_j -= torch.stack(history_j.values).squeeze().detach()

                    losses_v_i = []
                    losses_v_j = []

                    action_j = torch.tensor(
                        history_j.actions, dtype=torch.int64
                    ).detach()  # OK

                    for R_i, R_j, p_i, a_i, p_j, a_j in zip(
                        reward_i,
                        reward_j,
                        probs_i,
                        action_i,
                        probs_j,
                        action_j,
                    ):
                        log_prob_i = Categorical(p_i).log_prob(a_i)
                        log_prob_j = Categorical(p_j).log_prob(a_j)
                        losses_v_i.append(log_prob_i * R_i.detach() * log_prob_j)
                        losses_v_j.append(log_prob_i * R_j.detach() * log_prob_j)

                    total_loss_v_i = torch.stack(losses_v_i).sum()
                    total_loss_v_j = torch.stack(losses_v_j).sum()

                    for grads, param_i, param_j in zip(
                        gradients, actor_i.parameters(), actor_j.parameters()
                    ):
                        D1Ri = grad(
                            total_loss_v_i,
                            (param_i, param_j),
                            create_graph=True,
                        )
                        D1Rj = grad(
                            total_loss_v_j,
                            (param_i, param_j),
                            create_graph=True,
                        )

                        D2Rj = [
                            grad(g, param_i, create_graph=True)[0].view(-1)
                            for g in D1Rj[1].view(-1)
                        ]

                        D2Rj = torch.stack(
                            [D2Rj[x] for x, _ in enumerate(param_i.view(-1))]
                        )
                        naive_grad = D1Ri[0].view(-1)
                        second_order_grad = torch.matmul(D2Rj, D1Ri[1].view(-1))
                        lola_grad = (
                            naive_grad + self.second_order_lr * second_order_grad
                        )
                        grads += lola_grad.view_as(grads)

            self.all_gradients[agent_i] = gradients

    def epoch_finished(self, epoch: int, tag: str) -> None:
        print("epoch finished")
        self.preprocess()
        print("preprocess finished")
        super(LolaPG, self).epoch_finished(epoch, tag)

    def update_actor(self, agent_id: AgentID, gamma: float, returns) -> None:
        print("update bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        actor_net = self.actor_networks[agent_id]
        actor_net.optimizer.zero_grad()
        for params, lola_grad in zip(
            actor_net.parameters(), self.all_gradients[agent_id]
        ):
            params.grad = lola_grad
        actor_net.optimizer.step()
