from mate.utils import get_param_or_default
from mate.controllers.actor_critic import ActorCritic
import torch
from torch.distributions import Categorical
from torch.autograd import grad


class LOLA(ActorCritic):
    def __init__(self, params):
        super(LOLA, self).__init__(params)
        self.second_order_learning_rate = get_param_or_default(
            params, "second_order_learning_rate", 1
        )

    def preprocess(self):
        all_gradients = []
        for agent_i in range(self.nr_agents):
            obs_i, _, actions_i, _, reward_i, old_probs_i, _, _ = self.memories[
                agent_i
            ].get_training_data()
            probs_i = self.actor_nets[agent_i](obs_i)
            reward_i -= self.critic_nets[agent_i](obs_i).squeeze().detach()
            gradients = []
            for weights in self.actor_nets[agent_i].parameters():
                gradients.append(torch.zeros_like(weights))
            for j in range(self.nr_agents):
                if j != agent_i:
                    (
                        histories_j,
                        _,
                        actions_j,
                        _,
                        reward_j,
                        old_probs_j,
                        _,
                        _,
                    ) = self.memories[j].get_training_data()
                    probs_j = self.actor_nets[j](histories_j)
                    reward_j -= self.critic_nets[j](histories_j).squeeze().detach()
                    losses_V_i = []
                    losses_V_j = []
                    for (
                        RewardI,
                        RewardI,
                        ProbI,
                        OldProbI,
                        ActionI,
                        ProbJ,
                        OldProbJ,
                        ActionJ,
                    ) in zip(
                        reward_i,
                        reward_j,
                        probs_i,
                        old_probs_i,
                        actions_i,
                        probs_j,
                        old_probs_j,
                        actions_j,
                    ):
                        log_prob_i = Categorical(ProbI).log_prob(ActionI)
                        log_prob_j = Categorical(ProbJ).log_prob(ActionJ)
                        losses_V_i.append(log_prob_i * RewardI * log_prob_j)
                        losses_V_j.append(log_prob_i * RewardI * log_prob_j)
                    total_loss_V_i = torch.stack(losses_V_i).sum()
                    total_loss_V_j = torch.stack(losses_V_j).sum()

                    for grads, param_i, param_j in zip(
                        gradients,
                        self.actor_nets[agent_i].parameters(),
                        self.actor_nets[j].parameters(),
                    ):
                        D1Ri = grad(
                            total_loss_V_i, (param_i, param_j), create_graph=True
                        )
                        D1Rj = grad(
                            total_loss_V_j, (param_i, param_j), create_graph=True
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
                            naive_grad
                            + self.second_order_learning_rate * second_order_grad
                        )
                        grads += lola_grad.view_as(grads)

            all_gradients.append(gradients)
        return all_gradients

    def update_actor(
        self, agent_id, training_data, actor_net, critic_net, preprocessed_data
    ):
        actor_net.optimizer.zero_grad()
        for params, lola_grad in zip(
            actor_net.parameters(), preprocessed_data[agent_id]
        ):
            params.grad = lola_grad
        actor_net.optimizer.step()
