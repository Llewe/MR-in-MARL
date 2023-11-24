from pettingzoo import AECEnv
from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0

if __name__ == "__main__":
    env = MeltingPotCompatibilityV0(
        substrate_name="stag_hunt_in_the_matrix__arena", render_mode="human"
    )
    print(isinstance(env, AECEnv))
    observations, infos = env.reset()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.step(actions)
    env.close()


"""
prisoners_dilemma_in_the_matrix__arena,
stag_hunt_in_the_matrix__arena,
collaborative_cooking__asymmetric,
pure_coordination_in_the_matrix__arena,
coins,
hidden_agenda,
predator_prey__random_forest,
chemistry__two_metabolic_cycles_with_distractors,
rationalizable_coordination_in_the_matrix__arena,
daycare,
running_with_scissors_in_the_matrix__repeated,
bach_or_stravinsky_in_the_matrix__repeated,
pure_coordination_in_the_matrix__repeated,
collaborative_cooking__cramped,
coop_mining,
commons_harvest__closed,
collaborative_cooking__forced,
prisoners_dilemma_in_the_matrix__repeated,
factory_commons__either_or,
chemistry__two_metabolic_cycles,
territory__open,
gift_refinements,
allelopathic_harvest__open,
chemistry__three_metabolic_cycles,
stag_hunt_in_the_matrix__repeated,
predator_prey__orchard,
bach_or_stravinsky_in_the_matrix__arena,
paintball__king_of_the_hill,
predator_prey__alley_hunt,
territory__inside_out,
chicken_in_the_matrix__repeated,
running_with_scissors_in_the_matrix__one_shot,
paintball__capture_the_flag,
collaborative_cooking__circuit,
rationalizable_coordination_in_the_matrix__repeated,
collaborative_cooking__figure_eight,
clean_up,
commons_harvest__partnership,
collaborative_cooking__crowded,
running_with_scissors_in_the_matrix__arena,
territory__rooms,
fruit_market__concentric_rivers,
predator_prey__open,
boat_race__eight_races,
chemistry__three_metabolic_cycles_with_plentiful_distractors,
chicken_in_the_matrix__arena,
collaborative_cooking__ring,
externality_mushrooms__dense,
commons_harvest__open

"""
