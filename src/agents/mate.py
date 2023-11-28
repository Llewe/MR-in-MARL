import numpy
from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.agents.a2c import A2C
from src.config.ctrl_config import MateConfig


class Mate(A2C):
    mate_mode: MateConfig.Mode
    defect_mode: MateConfig.DefectMode
    token_value: float

    trust_request_matrix: numpy.ndarray
    trust_response_matrix: numpy.ndarray

    last_rewards_observed: dict[AgentID, list[float]]

    nr_agents: int

    agent_id_mapping: dict[AgentID, int]
    neighborhood: dict[AgentID, list[AgentID]]

    def __init__(self, config: MateConfig):
        super(Mate, self).__init__(config)

        self.mate_mode = config.MODE
        self.defect_mode = config.DEFECT_MODE
        self.token_value = config.TOKEN_VALUE

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super(Mate, self).init_agents(action_space, observation_space)
        self.nr_agents = len(action_space.keys())

        self.agent_id_mapping = {}
        self.neighborhood = {}

        for i, agent_id in enumerate(action_space.keys()):
            self.agent_id_mapping[agent_id] = i
            self.neighborhood[agent_id] = [
                x for x in action_space.keys() if x != agent_id
            ]

        self.trust_request_matrix = numpy.zeros(
            (self.nr_agents, self.nr_agents), dtype=numpy.float32
        )
        self.trust_response_matrix = numpy.zeros(
            (self.nr_agents, self.nr_agents), dtype=numpy.float32
        )

        self.last_rewards_observed = {agent_id: [] for agent_id in action_space.keys()}

    def epoch_started(self, epoch: int) -> None:
        super(Mate, self).epoch_started(epoch)
        self.last_rewards_observed = {
            agent_id: [] for agent_id in self.actor_networks.keys()
        }

    def epoch_finished(self, epoch: int) -> None:
        super(Mate, self).epoch_finished(epoch)

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        super(Mate, self).step_agent(
            agent_id, last_observation, last_action, reward, done
        )

    def step_finished(self, step: int) -> None:
        super(Mate, self).step_finished(step)
        self.mate(step)

    def can_rely_on(
        self, agent_id: AgentID, reward: float
    ):  # history, next_history is missing
        if self.mate_mode == MateConfig.Mode.STATIC_MODE:
            is_empty = self.last_rewards_observed[agent_id]
            if not is_empty:
                self.last_rewards_observed[agent_id].append(reward)
                return True
            last_reward = numpy.mean(self.last_rewards_observed[agent_id])
            self.last_rewards_observed[agent_id].append(reward)
            return reward >= last_reward
        if self.mate_mode == MateConfig.Mode.TD_ERROR_MODE:
            raise NotImplementedError
            # history = torch.tensor(
            #     numpy.asarray([history]), dtype=torch.float32, device=self.device
            # )
            # next_history = torch.tensor(
            #     numpy.asarray([next_history]), dtype=torch.float32, device=self.device
            # )
            # value = self.get_values(agent_id, history)[0].item()
            # next_value = self.get_values(agent_id, next_history)[0].item()
            # return reward + self.gamma * next_value - value >= 0
        if self.mate_mode == MateConfig.Mode.VALUE_DECOMPOSE_MODE:
            return False

    def mate(self, step: int):
        """
        V = approximated value function
        N = sum of all neighbors agents
        tau = history
        e = experience
        at = action
        st = state
        z = observation/local observation
        r = reward
        e = <tau,action,reward,observation> -> History zum Zeitpunkt i
        𝑒𝑡,𝑖 = ⟨𝜏𝑡,𝑖, 𝑎𝑡,𝑖, 𝑟𝑡,𝑖, 𝑧𝑡+1,𝑖 ⟩

        1: procedure MATE(MI𝑖, ˆ 𝑉𝑖, N𝑡,𝑖, 𝜏𝑡,𝑖, 𝑒𝑡,𝑖 )
        2:      𝑟req ← 0, ˆ 𝑟res ← 0
        3:      if MI𝑖 (𝑟𝑡,𝑖 ) ≥ 0 then
        4:          Send acknowledgment request 𝑥𝑖 = 𝑥token to all 𝑗 ∈ N𝑡,𝑖
        5:      for neighbor agent 𝑗 ∈ N𝑡,𝑖 do ⊲ Respond to requests
        6:          if request 𝑥 𝑗 received from 𝑗 then
        7:              𝑟req ← max{ ˆ 𝑟req, 𝑥 𝑗 }
        8:              if MI𝑖 (𝑟𝑡,𝑖 + 𝑥 𝑗 ) ≥ 0 then
        9:                  Send response 𝑦𝑖 = +𝑥 𝑗 to agent 𝑗
        10:             else
        11:                 Send response 𝑦𝑖 = −𝑥 𝑗 to agent 𝑗
        12:     if MI𝑖 (𝑟𝑡,𝑖 ) ≥ 0 then ⊲ If requests have been sent before
        13:         for neighbor agent 𝑗 ∈ N𝑡,𝑖 do ⊲ Receive responses
        14:             if response 𝑦𝑗 received from 𝑗 then
        15:                 𝑟res ← min{ ˆ 𝑟res, 𝑦𝑗 }
        16:     return 𝑟𝑡,𝑖 + ˆ 𝑟req + ˆ 𝑟res (ˆ 𝑟 MATE 𝑡,𝑖 as defined in Eq. 5)

        """
        if step == 1:
            for v in self.last_rewards_observed.values():
                v.clear()

        original_rewards: dict[AgentID, float] = {
            a: self.step_info[a].rewards[-1] for a in self.step_info.keys()
        }
        mate_rewards: dict[AgentID, float] = {
            a: self.step_info[a].rewards[-1] for a in self.step_info.keys()
        }

        # transition = super(MATE, self).prepare_transition(
        #     joint_histories, joint_action, rewards, next_joint_histories, done, info
        # )

        self.trust_request_matrix[:] = 0
        self.trust_response_matrix[:] = 0

        # 1. Send trust requests
        defector_id: int = -1

        if self.defect_mode != MateConfig.DefectMode.NO_DEFECT:
            defector_id = numpy.random.randint(0, self.nr_agents)

        for agent_id in self.agent_id_mapping.keys():
            i = self.agent_id_mapping[agent_id]
            reward = original_rewards[agent_id]

            # for i, reward, history, next_history in zip(
            #     range(self.nr_agents),
            #     original_rewards,
            #     joint_histories,
            #     next_joint_histories,
            # ):
            requests_enabled = i != defector_id or self.defect_mode not in [
                MateConfig.DefectMode.DEFECT_ALL,
                MateConfig.DefectMode.DEFECT_SEND,
            ]

            if requests_enabled and self.can_rely_on(
                agent_id, reward
            ):  # Analyze the "winners" of that step
                for neighbor in self.neighborhood[agent_id]:
                    j = self.agent_id_mapping[neighbor]
                    self.trust_request_matrix[j][i] += self.token_value
                    # transition["request_messages_sent"] += 1 logging

        # 2. Send trust responses
        for agent_id in self.agent_id_mapping.keys():
            i = self.agent_id_mapping[agent_id]
            # for i, history, next_history in zip(
            #     range(self.nr_agents), joint_histories, next_joint_histories
            # ):
            respond_enabled = i != defector_id or self.defect_mode not in [
                MateConfig.DefectMode.DEFECT_ALL,
                MateConfig.DefectMode.DEFECT_RESPONSE,
            ]

            trust_requests = [
                self.trust_request_matrix[i][self.agent_id_mapping[j]]
                for j in self.neighborhood[agent_id]
            ]
            if len(trust_requests) > 0:
                mate_rewards[agent_id] += numpy.max(trust_requests)

            if respond_enabled and len(self.neighborhood[agent_id]) > 0:
                if self.can_rely_on(agent_id, mate_rewards[agent_id]):
                    accept_trust = self.token_value
                else:
                    accept_trust = -self.token_value
                for neighbor in self.neighborhood[agent_id]:
                    j = self.agent_id_mapping[neighbor]
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust
                        # if accept_trust > 0:
                        #     transition["response_messages_sent"] += 1
        # 3. Receive trust responses
        for agent_id in self.agent_id_mapping.keys():
            i = self.agent_id_mapping[agent_id]
            trust_responses = self.trust_response_matrix[i]

            receive_enabled = i != defector_id or self.defect_mode not in [
                MateConfig.DefectMode.DEFECT_ALL,
                MateConfig.DefectMode.DEFECT_RESPONSE,
            ]

            if (
                receive_enabled
                and len(self.neighborhood[agent_id]) > 0
                and trust_responses.any()
            ):
                filtered_trust_responses = [
                    trust_responses[self.agent_id_mapping[x]]
                    for x in self.neighborhood[agent_id]
                    if abs(trust_responses[self.agent_id_mapping[x]]) > 0
                ]
                if len(filtered_trust_responses) > 0:
                    mate_rewards[agent_id] += min(filtered_trust_responses)

        #  write mate rewards to step_info
        for agent_id in self.agent_id_mapping.keys():
            self.step_info[agent_id].rewards[-1] = mate_rewards[agent_id]
