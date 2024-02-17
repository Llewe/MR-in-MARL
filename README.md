# MR-in-MARL
This is the repository for my master thesis on "Beeinflussung von Verhalten durch Reward-Manipulation im Multi-Agent Reinforcement-Learning".

There are three main parts of the repository:
1. Code to run experiments
2. Code to replay experiments
3. Code to convert TensorBoard logs to Mathplotlib plots


## Setup
You need Python 3.10.13 (other 3.10 Versions will probably also work) to run the code.
And Poetry to install the dependencies.

```bash
# Create Poetry environment with Python 3.10
poetry env use 3.10

# Install dependencies (if you want mypy and pytest add --with dev to the install command)
poetry install

# Activate the environment
poetry shell
```
Im using Pyadntic settings to configure everything. All possible settings classes are in the Package `src.config`.
To change settings you can either modify the default values inside the settings classes or run the code with the desired changes as environment variables.
I would recommend the second approach, because it is more flexible and you can easily run the code with different settings (especially when running the experiments with scripts).

## Run Experiments
To start a new experiment run `src/training.py' with the desired settings as environment variables.
For example:
```sh
# Run the training with the default settings
python src/training.py
```
All the logs will be saved in the `resources` folder. The logs are saved in the TensorBoard format and can be visualized with TensorBoard.

A script how to run the experiments is the `trainings.sh` file. Just add the desired experiments to the file and run it with the following command:
```sh
sh trainings.sh
```
There are some examples inside the file. Be aware that I run multiple experiments in parallel. Every enabled experiment in this file will be run in parallel. So you need to find a good balance between the number of experiments and the number of available CPU cores.

## Show TensorBoard logs
Note in which directory the logs are saved and run the following command in the terminal:
```bash
tensorboard --logdir <path_to_logs> # e.g. tensorboard --logdir resources/coin_game/default"
```
Then open the link printed in the terminal in your browser (usually `http://localhost:6006/`).

## Replay Experiments
To replay an experiment run `src/replay.py` with the desired settings as environment variables.
This can be useful to visualize and watch the behavior of the agents in the environment. For that you need to set the correct `replay` settings.
Do not forget to set the visualization settings to `human` to see a rendered window ;).

## Convert TensorBoard logs to Mathplotlib plots
To convert the TensorBoard logs to Mathplotlib plots run `diagram_gen` with the desired settings as environment variables.
This the code here is not very flexible if you run other environments than the coin game you need to modify the code.
```sh 
python src/diagram_gen.py
```
The plots will be saved in the `output` folder.


## Some code details
- The Agents are implemented in "src.controllers". Each Controller implements an interface and can be used to train and to replay the agents.
- All manipulation parts are implemented in the "src.controller_ma" package. Since the manipulation is separated from the agent itself its implementation in a separate controller. In the Training loop its ensured that the manipulation controller will receive the correct observations (global or local) and that the manipulation is applied to the agents.
- Since some PettingZoo environments do habe differnt observation spaces, the observation spaces, not all environments are supported. To enable support you probably need flatten the observation space in the environment.
- MATE is only "Partially implemented" since i didn't used the Harvest environment in the final tests, the neighbors was not used. The code prepares the use of neighbors but currently every agent sees all the other agents.