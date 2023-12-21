#!/bin/bash
log_counter=1
train() {
    local log_file="./resources/logs/$(date -d "today" "+%Y.%m.%d-%H.%M.%S"_$log_counter).log"
    ((log_counter++))

    local command="poetry run python src/training.pyx"

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
python3 setup.py build_ext --inplace

#train AGENT_TYPE=random EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000"
#train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.0001 CRITIC_LR=0.0001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 ACTOR_LR=0.0001 CRITIC_LR=0.0001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=0.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=0.8 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=1 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=1.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
#
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=0.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=0.8 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=1 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=1.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
#train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_my_coin_game ENV_TAG="4pl-50000" DISCOUNT_FACTOR=0.95 TOKEN_VALUE=2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1



train AGENT_TYPE=random EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000"
train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.0001 CRITIC_LR=0.0001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
train AGENT_TYPE=a2c EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 ACTOR_LR=0.0001 CRITIC_LR=0.0001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1

train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=0.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=0.8 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=1 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=1.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=2e-05 EPSILON_MIN=0.1

train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=0.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=0.8 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=1 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=1.2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1
train AGENT_TYPE=mate EPOCHS=50000 ENV_NAME=p_harvest ENV_TAG="50000" DISCOUNT_FACTOR=0.99 TOKEN_VALUE=2 ACTOR_LR=0.001 CRITIC_LR=0.001 EPSILON_INIT=0.8 EPSILON_DECAY=4.66e-05 EPSILON_MIN=0.1