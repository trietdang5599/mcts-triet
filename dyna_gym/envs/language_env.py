from collections import OrderedDict

import gym
import torch


class LanguageEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """
    def __init__(self, terminal_token, horizon, reward_func):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        self.get_reward = reward_func
        self.state = None
        self.input_len = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        input_ids = options.get("input_ids")
        attention_mask = options.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        self.state = (input_ids, attention_mask)
        self.input_len = len(input_ids)
        # API mới: trả về (obs, info)
        return self.state, {}

    def transition(self, s, a, is_model_dynamic=False):
        ids, attention_mask = s
        next_ids = torch.cat([ids, torch.tensor([a]).to(ids.device)])
        attention_mask = torch.cat([attention_mask, torch.tensor([1]).to(attention_mask.device)])

        done = (a == self.terminal_token) or (len(next_ids) == self.horizon)

        reward = self.get_reward((next_ids, attention_mask)) if done else 0
        return (next_ids, attention_mask), reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)
        return self.state, reward, done, {}  # 4-tuple cho Gym cũ


    def equality_operator(self, s1, s2):
        # s1 and s2 are two tensors
        return all(torch.equal(x1, x2) for x1, x2 in zip(s1, s2))
