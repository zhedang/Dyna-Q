# Flappy Bird RL

This project implements a Q-learning agent to play the game of Flappy Bird.

## Description

The agent learns to play Flappy Bird by using a Q-learning algorithm. The state is a simplified representation of the game, and the agent learns a policy to maximize its reward.

The state is defined by:
*   The bird's y-position.
*   The bird's velocity.
*   The y-position of the next pipe.
*   The x-position of the next pipe.

The agent can perform two actions:
*   Jump
*   Do nothing

## Demo

[Watch a demo of the agent in action.](https://drive.google.com/file/d/1jzxIh3ZthdDOpsgRCTUaoyMLqt2GTWZJ/view?usp=share_link)

## How to Run

To run the game, execute the following command:

```bash
python flappy_bird_rl/main.py
```

## Configuration

The game and agent parameters can be configured in `flappy_bird_rl/config.py`.

### Game Settings

*   `SCREEN_WIDTH`, `SCREEN_HEIGHT`: Dimensions of the game window.
*   `FPS`: Frames per second.
*   `PIPE_GAP_SIZE`: The vertical gap between the upper and lower pipes.
*   `PIPE_SPACING`: The horizontal distance between pipes.

### Agent Settings

*   `LEARNING_RATE`: The learning rate for the Q-learning algorithm.
*   `DISCOUNT_FACTOR`: The discount factor for future rewards.
*   `EPSILON_START`, `EPSILON_END`, `EPSILON_DECAY`: Parameters for the epsilon-greedy policy.
