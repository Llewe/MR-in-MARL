export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0 EPSILON_INIT=0.5 ACTOR_LR=0.0001 CRITIC_LR=0.0001 AGENT_TAG=discount_0_lr_0.0001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0.95 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.0001 CRITIC_LR=0.0001 AGENT_TAG=discount_0.95_reward_norm_0_lr_0.0001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0.5 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.0001 CRITIC_LR=0.0001 AGENT_TAG=discount_0.5_reward_norm_0_lr_0.0001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.0001 CRITIC_LR=0.0001 AGENT_TAG=discount_0_reward_norm_0_lr_0.0001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0.95 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.001 CRITIC_LR=0.001 AGENT_TAG=discount_0.95_reward_norm_0_lr_0.001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0.5 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.001 CRITIC_LR=0.001 AGENT_TAG=discount_0.5_reward_norm_0_lr_0.001_eps_0.5
poetry run python src/training.py
export AGENT_TYPE=actor_critic EPISODES=10000 DISCOUNT_FACTO=0 REWARD_NORMALIZATION=0 EPSILON_INIT=0.5 ACTOR_LR=0.001 CRITIC_LR=0.001 AGENT_TAG=discount_0_reward_norm_0_lr_0.001_eps_0.5
poetry run python src/training.py
