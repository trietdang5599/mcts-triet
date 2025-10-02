import gym
import dyna_gym.agents.uct as uct
from gym.utils import seeding
### Parameters
# Tạo env theo Gym mới (render_mode nếu muốn cửa sổ)
env = gym.make('NSCartPole-v0')

agent = uct.UCT(action_space=env.action_space, rollouts=100)
timesteps = 100
verbose = False



def reset(self, *, seed: int | None = None, options: dict | None = None):
    if seed is not None:
        self.np_random, _ = seeding.np_random(seed)
    # TODO: khởi tạo trạng thái như code cũ
    # self.state = ...
    obs = self._get_obs()  # hoặc ghép từ self.state
    return obs, {}

def step(self, action):
    # TODO: cập nhật state, reward theo code cũ
    terminated = bool(done_condition)        # ví dụ: pole đổ
    truncated  = bool(time_limit_condition)  # ví dụ: vượt timesteps
    info = {}
    obs = self._get_obs()
    return obs, reward, terminated, truncated, info


### Run
obs, info = env.reset()
done = False

for ts in range(timesteps):
    action = agent.act(env, done)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if verbose:
        # env.print_state() nếu env có; nếu không thì bỏ
        pass
    # Gym mới render đã dựa vào render_mode; gọi env.render() có thể không cần

    if done:
        print(f"Episode finished after {ts+1} timesteps")
        break
else:
    print(f"Successfully reached end of episode ({timesteps} timesteps)")