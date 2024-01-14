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
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=random NAME="Zufall" EXP_TAG="Zufall"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 1"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 2"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 1" TOKEN_VALUE=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 2" TOKEN_VALUE=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 3" TOKEN_VALUE=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 1 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 2 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 1 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 2 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 1 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 2 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 1"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 2"
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 1 - [0.8]" PERCENTAGE=0.8
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 2 - [0.8]" PERCENTAGE=0.8
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 1 - [1.0]" PERCENTAGE=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 2 - [1.0]" PERCENTAGE=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
train EPOCHS=10000 ENV_NAME=p_coin_game PLAYERS=2 GRID_SIZE=3 ENV_TAG="final-10000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5


#################################
# Coin-Game 4
#
# ENV_TAG="4pl-5000" PLAYERS=4 GRID_SIZE=5
#################################

#train AGENT_TYPE=random NAME="Zufall" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001
#train AGENT_TYPE=actor_critic NAME="Actor-Critic" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95
#train AGENT_TYPE=mate NAME="MATE-TD" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Static" MODE="static" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Value-Decompose" MODE="value_decompose" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=gifting NAME="Gifting-ZS" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-FB (10)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="fixed_budget" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-RB" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="replenishable_budget" ACTION_MODE="random"
#train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=0.5)"  EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=0.5)"  EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=1.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.5
#train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0


#################################
# IPD
#################################
#
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=random NAME="Zufall" EXP_TAG="Zufall"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 1"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=actor_critic NAME="Actor-Critic" EXP_TAG="Actor-Critic 2"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 1" TOKEN_VALUE=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 2" TOKEN_VALUE=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=mate NAME="MATE-TD" EXP_TAG="MATE-TD 3" TOKEN_VALUE=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 1 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [0.5]" EXP_TAG="Gifting-ZS 2 - [0.5]"  GIFT_REWARD=0.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 1 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1]" EXP_TAG="Gifting-ZS 2 - [1]"  GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 1 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=none AGENT_TYPE=gifting NAME="Gifting-ZS [1.5]" EXP_TAG="Gifting-ZS 2 - [1.5]"  GIFT_REWARD=1.5 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 1"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_heuristic AGENT_TYPE=actor_critic NAME="Heuristik" EXP_TAG="Heuristik 2"
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 1 - [0.8]" PERCENTAGE=0.8
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [0.8]" EXP_TAG="Zentraler Prozentsatz 2 - [0.8]" PERCENTAGE=0.8
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 1 - [1.0]" PERCENTAGE=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_fixed_percentage AGENT_TYPE=actor_critic NAME="Zentraler Prozentsatz - [1.0]" EXP_TAG="Zentraler Prozentsatz 2 - [1.0]" PERCENTAGE=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [-0.5-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-0.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.0]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=central_ac_percentage AGENT_TYPE=actor_critic NAME="Zentrale AC-Strafe - [0-1.5]" EXP_TAG="Zentrale AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-0.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.0]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_percentage AGENT_TYPE=actor_critic NAME="Individuelle AC-Strafe - [0-1.5]" EXP_TAG="Individuelle AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [-0.5-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [-0.5-0.5]" LOWER_BOUND=-0.5 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-0.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-0.5]" LOWER_BOUND=0 UPPER_BOUND=0.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.0]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.0]" LOWER_BOUND=0 UPPER_BOUND=1.0
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 1 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5
#train EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="final-5000" DISCOUNT_FACTOR=0.95 MA_MODE=individual_ac_p_global_metric AGENT_TYPE=actor_critic NAME="Individuelle Metric AC-Strafe - [0-1.5]" EXP_TAG="Individuelle Metric AC-Strafe 2 - [0-1.5]" LOWER_BOUND=0 UPPER_BOUND=1.5

#################################
# IPD - Punish Defect token value
#################################
#
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=0.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=1)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=1.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=2)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=2.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=3)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=3.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=3.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=3.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-0.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-1)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-1.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-2)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-2.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-3)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-3.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP-PD (x=-3.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="rmp-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-3.5
#################################
# IPD - SW token value
#################################
#
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=0.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=1)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=1.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=2)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=2.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=3)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=3.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=3.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=3.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-0.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-1)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-1.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-2)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-2.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.5
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-3)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-3.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-3.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="sw-pd-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-3.5



#################################
# HARVEST
#################################

#train AGENT_TYPE=random NAME="Zufall" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.001 CRITIC_LR=0.001
#train AGENT_TYPE=actor_critic NAME="Actor-Critic" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=actor_critic NAME="Actor-Critic" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="Mate-TD" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0 MODE="td_error"
#train AGENT_TYPE=mate NAME="Mate-TD" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0 MODE="td_error"
#train AGENT_TYPE=ma_harvest_optimal_path_ac NAME="Harvest-ToTheWall (x=1)" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_harvest_optimal_path_ac NAME="Harvest-ToTheWall (x=2)" EPOCHS=15000 ENV_NAME=p_harvest ENV_TAG="default-15000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=2.0