
import pygame
from game_env import FlappyBirdEnv
from agent import QLearningAgent
import config

def main():
    env = FlappyBirdEnv()
    agent = QLearningAgent()
    num_episodes = 20000 # Increased episodes for better learning

    print("Starting training with Agent Monitor...")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 1. Get agent's internal state before choosing action
            discrete_state = agent._discretize_state(state)
            q_values = agent.q_table[discrete_state]
            
            # 2. Choose action
            action = agent.choose_action(state)
            
            # 3. Package debug info
            debug_info = {
                'episode': episode + 1,
                'state': discrete_state,
                'q_values': q_values,
                'action': action,
                'epsilon': agent.epsilon
            }

            # 4. Pass everything to the environment step
            next_state, reward, done = env.step(action, debug_info)
            
            # 5. Agent learns from the result
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Score: {env.score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    print("Training finished.")
    pygame.quit()

if __name__ == "__main__":
    main()
