from datetime import datetime
from typing import Callable, Sequence

import gym
import torch
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.hf_default_policy import HuggingFaceDefaultPolicy
from dyna_gym.utils.tree_search_utils import print_tree


def uct_for_hf_transformer_pipeline(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        horizon: int = 100,
        reward_func: Callable | None = None,
        uct_args: dict = {},
        model_generation_args: dict = {},
        should_plot_tree: bool = False,
        should_print_tree: bool = True,
        reward_func_input_is_state: bool = False,
        decode_skip_special_tokens: bool = True,
) -> Callable:
    """
    A wrapped UCT agent for HuggingFace transformer.

    Args:
        model_name: The name of a HuggingFace transformer model. If provided, will load the model and tokenizer.
        model: A HuggingFace transformer model.
        tokenizer: A HuggingFace tokenizer.
        horizon: The maximum number of steps to take.
        reward_func: A function that evaluate the reward of a sequence.
        value_func: A function that evaluate the value of a sequence.
        uct_args: Arguments for the UCT agent.
        model_generation_args: Arguments for the model generation.
        should_plot_tree: Whether to plot the tree after generation.
        should_print_tree: Whether to print the tree structure to stdout after generation.
        reward_func_input_is_state: Whether the input of the reward function is (token ids, attention masks) or tokenized text.
        decode_skip_special_tokens: Whether decoded outputs should skip special tokens by default.
    """
    if reward_func is None:
        reward_func = lambda _: 0.0  # type: ignore[assignment]

    eos_token_id = tokenizer.eos_token_id

    if not reward_func_input_is_state:
        # by default reward function takes tokenized text as input
        # if reward function takes texts as input, wrap it here to take (token ids, attention masks) as input
        def reward_func_(state):
            ids, attention_mask = state
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return reward_func(text)
    else:
        reward_func_ = reward_func

    env = gym.make(
        'LanguageEnv-v0',
        terminal_token=eos_token_id,
        horizon=horizon,
        reward_func=reward_func_,
    )

    default_policy = HuggingFaceDefaultPolicy(
        env=env,
        horizon=horizon,
        model=model,
        generation_args=model_generation_args,
    )

    agent = uct.UCT(
        default_policy=default_policy,
        **uct_args
    )

    default_reward = reward_func_
    default_skip_special_tokens = decode_skip_special_tokens

    ### Run
    def generate(
            input_ids=None,
            input_str=None,
            attention_mask=None,
            reward_override: Callable | None = None,
            skip_special_tokens: bool | None = None,
    ):
        assert (input_ids is None) != (input_str is None), "Only provide one of input_ids and input_str"

        if input_str is not None:
            input_ids = tokenizer.encode(input_str)
            input_ids = torch.tensor(input_ids).to(model.device)

        if attention_mask is None:
            # attend to tokens that are not padding
            if tokenizer.pad_token_id is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            attention_mask = attention_mask.to(model.device)

        if reward_override is not None:
            if reward_func_input_is_state:
                env.get_reward = reward_override
            else:
                def wrapped_reward(state):
                    ids, attention_mask = state
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    return reward_override(text)

                env.get_reward = wrapped_reward
        else:
            env.get_reward = default_reward

        # Gọi reset với options, không truyền positional args
        env.reset(options={"input_ids": input_ids, "attention_mask": attention_mask})

        # step trả 5 phần tử, nhưng bạn đang không unpack => vẫn OK.
        env.step(agent.act(env, done=False))

        if should_print_tree:
            print_tree(agent.root, tokenizer)
        # optionally, plot the tree and save to a pdf file
        if should_plot_tree:
            # plot (and print) the tree
            from dyna_gym.utils.tree_search_utils import plot_tree
            filename = f"tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            plot_tree(agent.root, tokenizer, filename)
            print(f"Tree plotted and saved to {filename}.pdf")

        skip_special = default_skip_special_tokens if skip_special_tokens is None else skip_special_tokens

        decoded_plain = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in agent.rolled_out_trajectories
        ]
        decoded_with_special = [
            tokenizer.decode(ids, skip_special_tokens=False)
            for ids in agent.rolled_out_trajectories
        ]

        results = {
            'output_ids': agent.rolled_out_trajectories,
            'rewards': agent.rolled_out_rewards,
            'texts': decoded_plain if skip_special else decoded_with_special,
            'texts_plain': decoded_plain,
            'texts_with_special_tokens': decoded_with_special,
        }

        # clear for the next generation call
        agent.reset()
        env.get_reward = default_reward

        return results

    return generate
