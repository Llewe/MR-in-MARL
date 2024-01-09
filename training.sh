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

#train AGENT_TYPE=random EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001
#train AGENT_TYPE=actor_critic EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_coin_to_middle EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=ma_coin_to_middle EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="2pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=2.0

#################################
# Coin-Game 4
#
# ENV_TAG="4pl-5000" PLAYERS=4 GRID_SIZE=5
#################################

train AGENT_TYPE=random NAME="Zufall" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001
train AGENT_TYPE=actor_critic NAME="Actor-Critic" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95
train AGENT_TYPE=mate NAME="MATE-TD" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Static" MODE="static" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Value-Decompose" MODE="value_decompose" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
train AGENT_TYPE=gifting NAME="Gifting-Zero-Sum" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-Fixed-Budget (10)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="fixed_budget" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-Replenishable-Budget" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="replenishable_budget" ACTION_MODE="random"
train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=-0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_coin_to_middle_ac NAME="RMP-MID (AC, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=-0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_coin_to_middle_mate NAME="RMP-MID (MATE-TD, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=-0.5)"  EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_social_wellfare_mate NAME="RMP-SW (MATE-TD, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=-0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (AC, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=-0.5)"  EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_dont_take_others_mate NAME="RMP-DTOC (MATE-TD, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=-0.5)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=-1)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
train AGENT_TYPE=ma_dont_take_others_ac NAME="RMP-DTOC (AC, x=-2)" EPOCHS=5000 ENV_NAME=p_coin_game ENV_TAG="4pl-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0


#################################
# IPD
#################################
#
#train AGENT_TYPE=random NAME="Zufall" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95
#train AGENT_TYPE=actor_critic NAME="Actor-Critic" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95
#train AGENT_TYPE=mate NAME="MATE-TD" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Static" MODE="static" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate NAME="MATE-Value-Decompose" MODE="value_decompose" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=gifting NAME="Gifting-Zero-Sum" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="zero_sum" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-Fixed-Budget (10)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="fixed_budget" ACTION_MODE="random"
#train AGENT_TYPE=gifting NAME="Gifting-Replenishable-Budget" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 GIFT_REWARD=1.0 GIFT_MODE="replenishable_budget" ACTION_MODE="random"
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP (x=-0.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-0.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP (x=-1.)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP (x=-1.5)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.5
#train AGENT_TYPE=ma_ipd_punish_defect NAME="RMP (x=-2.0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-1.0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-1.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-2.0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-2.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-3.0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-3.0
#train AGENT_TYPE=ma_social_wellfare_ac NAME="RMP-SW (x=-4.0)" EPOCHS=5000 ENV_NAME=p_prisoners_dilemma ENV_TAG="default-5000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 MANIPULATION_AMOUNT=-4.0


#################################
# HARVEST
#################################

#train AGENT_TYPE=random EPOCHS=5000 ENV_NAME=p_harvest ENV_TAG="default-5000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.001 CRITIC_LR=0.001
#train AGENT_TYPE=actor_critic EPOCHS=5000 ENV_NAME=p_harvest ENV_TAG="default-5000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=actor_critic EPOCHS=5000 ENV_NAME=p_harvest ENV_TAG="default-5000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_harvest ENV_TAG="default-5000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0 MODE="td_error"
#train AGENT_TYPE=mate EPOCHS=5000 ENV_NAME=p_harvest ENV_TAG="default-5000" DISCOUNT_FACTOR=0.99 MANIPULATION_AMOUNT=1.0 MODE="td_error"