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
}
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=1
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=0.2
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=2
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=1 ACTOR_LR=0.0001 CRITIC_LR=0.0001
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=0.2 ACTOR_LR=0.0001 CRITIC_LR=0.0001
train AGENT_TYPE=mate ENV_NAME=my_coin_game TOKEN_VALUE=2 ACTOR_LR=0.0001 CRITIC_LR=0.0001
