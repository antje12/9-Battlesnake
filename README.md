# 9-battlesnake

## Folder overview
- /a-star-baseline: an A* agent for playing Battlesnake
- /battlesnake-server: the Battlesnake server code for local execution
- /dqn-learning-python-ai: the DQN agent training application
- /dqn-learning-python-ai/battlesnake_gym: the OpenAI Battlesnake gym
- /dqn-learning-python-ai/final model: the final DQN model and console output for documentation
- /dqn-learning-python-ai/final test: the 1000 test runs of DQN and A*
- /dqn-playing-python-ai: the DQN agent playing application

## Agent training
- Go to the /dqn-learning-python-ai folder
- Run "pip install -r requirements.txt"
- Run "python main.py"
- Copy the "final_model.pth" file

## Agent execution
To run the game locally 2 consoles are needed

### Console 1 - The AI server
- Go to the /dqn-playing-python-ai folder
- Run "pip install -r requirements.txt"
- Insert the "final_model.pth" file
- Run "python main.py"

### Console 2 - The game engine
- Go to the /battlesnake-server folder
- Run "single-run.cmd" or "multi-run.cmd" depending on a single- or multi-agent run with a console representation
- - Alternative "-browser" versions exist for a more visual representation akin to the official Battlesnake UI