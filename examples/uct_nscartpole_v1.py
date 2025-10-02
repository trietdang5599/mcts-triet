import gym
import dyna_gym.agents.uct as uct

### Parameters
env = gym.make('NSCartPole-v1')
agent = uct.UCT(
    action_space=env.action_space,
    rollouts=500,
    horizon=200,
    is_model_dynamic=True
)
timesteps = 100
verbose = False

### Run
env.reset()
done = False
for ts in range(timesteps):
    __, reward, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    env.render()
    if ts+1 == timesteps:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
