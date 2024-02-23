#!/bin/bash
log_counter=1
train() {
    local log_file="./resources/logs/$(date -d "today" "+%Y.%m.%d-%H.%M.%S"_$log_counter).log"
    ((log_counter++))

    local command="poetry run python src/training.py"

    # Loop through the provided parameters and export them
    for var in "$@"; do
        export "$var"
        command+=" $var"
    done

    # Run the command with the specified environment variables
    nohup $command > "$log_file" 2>&1 &

    # Unset the environment variables after the command has run
    for var in "$@"; do
        unset "$var"
    done
    sleep .5
}
poetry install --sync --all-extras



#################################
# Coin-Game 2
#
# ENV_TAG="2pl-5000" PLAYERS=2 GRID_SIZE=3
#################################
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=random NAME="Zufall" EXP_TAG="Zufall"
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 1" REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 2" REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 1" TOKEN_VALUE=1.0 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 2" TOKEN_VALUE=1.0 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 3" TOKEN_VALUE=1.0 REST_SEED=3
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 1 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 2 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 1 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 2 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 1 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 2 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random" REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 1" REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 2" REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 1 - [0.8]" PERCENTAGE=0.8 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 2 - [0.8]" PERCENTAGE=0.8 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 1 - [1.0]" PERCENTAGE=1.0 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 2 - [1.0]" PERCENTAGE=1.0 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=2
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=1
train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0 REST_SEED=2
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=1
#train EPOCHS=5000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="sfinal-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5 REST_SEED=2


#################################
# Coin-Game 4
#
# ENV_TAG="4pl-5000" PLAYERS=4 GRID_SIZE=5
#################################
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=random NAME="Zufall" EXP_TAG="Zufall"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 1"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 2"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 1" TOKEN_VALUE=1.0
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 2" TOKEN_VALUE=1.0
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 3" TOKEN_VALUE=1.0
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 1 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 2 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 1 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 2 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 1 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 2 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 1"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 2"
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-2] D5" EXP_TAG="Zentrale AC-Strafe 1 - [0-2]" LOWER_BOUND=0.0 UPPER_BOUND=2.0 UPDATE_EVERY_X_EPISODES=5
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-2] D5" EXP_TAG="Zentrale AC-Strafe 2 - [0-2]" LOWER_BOUND=0.0 UPPER_BOUND=2.0 UPDATE_EVERY_X_EPISODES=5
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-2] D5" EXP_TAG="Individuelle AC-Strafe 1 - [0-2]" LOWER_BOUND=0.0 UPPER_BOUND=2.0 UPDATE_EVERY_X_EPISODES=5
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-2] D5" EXP_TAG="Individuelle AC-Strafe 2 - [0-2] " LOWER_BOUND=0.0 UPPER_BOUND=2.0 UPDATE_EVERY_X_EPISODES=5
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-2] D5" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-2]" LOWER_BOUND=0.0 UPPER_BOUND=2.0  UPDATE_EVERY_X_EPISODES=5
#train ACTOR_LR=0.0005 CRITIC_LR=0.0005 EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=4 GRID_SIZE=5 ENV_TAG="4pl-final-10000-lr0005" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-2] D5" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-2]" LOWER_BOUND=0.0 UPPER_BOUND=2.0 UPDATE_EVERY_X_EPISODES=5