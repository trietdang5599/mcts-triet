import gym

from dyna_gym.agents import mcts

# Parameters
# Add render_mode='human' to visualize if the environment supports it.
env = gym.make('NSCartPole-v0')
agent = mcts.MCTS(action_space=env.action_space, rollouts=100)
timesteps = 100
verbose = False

# Run episode
obs, info = env.reset()
done = False

for ts in range(1, timesteps + 1):
    action = agent.act(env, done)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if verbose and hasattr(env, 'print_state'):
        env.print_state()

    if done:
        print(f"Episode finished after {ts} timesteps")
        break
else:
    print(f"Successfully reached end of episode ({timesteps} timesteps)")

env.close()
