from .mr_agent import MRAgent
from pettingzoo.utils.env import AgentID, ObsType, ActionType
import numpy as np


class DemoManipulationAgent(MRAgent):
    manipulation_amount = 0.1

    def __init__(self, *args, **kwargs):
        self.set_callback("agent_0", self._man_agent_0)
        super().__init__(*args, **kwargs)

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        # check if we wanna manipulate this agent
        if agent_id == "agent_1":
            agent_offset = 0
            # Agent observation space:
            # [
            # self_pos 0:1 ,
            # self_vel 2:3 ,
            # goal_rel_position 4:5,
            # landmark_rel_position 6:6+(N-1),
            # other_agent_rel_positions 6+N:6*2N]

            # We have a naive and simple plan: Try to send the other agents to the other goals
            # -> Agent 0 to goal, Agent 1 to landmark 1, Agent 2 to landmark 2...
            # agent location
            pos_agent = np.array([last_obs[0], last_obs[1]])
            vel_agent = np.array([last_obs[2], last_obs[3]])

            pos_fake_target_lm = np.array(
                [last_obs[6 + agent_offset * 2], last_obs[7 + agent_offset * 2]]
            )
            # no we need to figure out if the agent after that action is closer to the fake target or not
            # process discrete action
            vel_update = np.array([0.0, 0.0])
            match last_act:
                case 1:
                    vel_update = np.array([-5.0, 0.0])
                case 2:
                    vel_update = np.array([5.0, 0.0])
                case 3:
                    vel_update = np.array([0.0, -5.0])
                case 4:
                    vel_update = np.array([0.0, 5.0])

            # now lets check if

            # Calculate direction vector to the target
            direction_to_target = pos_fake_target_lm - pos_agent

            # Normalize direction vector
            direction_to_target_unit = direction_to_target / np.linalg.norm(
                direction_to_target
            )

            # Normalize current velocity
            vel_agent_unit = vel_agent / np.linalg.norm(vel_agent)

            # Calculate dot product with normalized current velocity
            dot_product = np.dot(vel_agent_unit, direction_to_target_unit)

            # Calculate new velocity by adding changing velocity
            new_velocity = vel_agent + vel_update

            # Normalize new velocity
            new_velocity_unit = new_velocity / np.linalg.norm(new_velocity)

            # Calculate dot product with normalized new velocity
            new_dot_product = np.dot(new_velocity_unit, direction_to_target_unit)

            # Check if the dot product is greater than 0
            if dot_product > 0 and new_dot_product > 0:
                # agent was and is still moving to the target
                return 0.0

            elif dot_product > 0:
                # agent was moving to the target, but is now moving away from it
                # punish the agent
                return -np.abs(reward) * self.manipulation_amount
            elif new_dot_product > 0:
                # agent was moving away from the target, but is now moving towards it
                # reward the agent
                return np.abs(reward) * self.manipulation_amount
            else:
                # agent was and is still moving away from the target

                if new_dot_product > dot_product:
                    # agent direction is improving
                    # reward the agent
                    return np.abs(reward) * self.manipulation_amount
                else:
                    # agent direction is getting worse
                    # punish the agent
                    return -np.abs(reward) * self.manipulation_amount

        return 0.0
