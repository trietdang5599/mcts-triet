# Monte-Carlo Tree Search for Large Language Models

This repository is a fork of [Dyna Gym](https://github.com/SuReLI/dyna-gym) and extends its functionality to focus on using Monte-Carlo tree search for decoding large language models (LLMs).

## Installation

First, create a new Conda environment (optional):

```bash
conda create --name mcts-for-llm python=3.10
conda activate mcts-for-llm
```
We tested on python 3.10. Other versions may work as well.

Then, git clone this repo and install the package:

```bash
pip install -e .
```

## Examples

### Using GPT-2 and UCT for Language Alignment with Positive Sentiment Reward

Run the following command to generate texts using the GPT-2 model, guided by UCT (Upper Confidence Bound applied to Trees) for language alignment. Positive sentiment is used as the reward.

```bash
python examples/uct_language_alignment.py
```

### Craigslist Bargains Negotiation Pipeline

1. **Supervised fine-tuning**. Train a small LLM on the raw Craigslist Bargains negotiations that live under `dataset/craigslist_bargains`:

```bash
python examples/train_craigslist_sft.py \
  --model_name distilgpt2 \
  --output_dir outputs/craigslist-sft
```

2. **MCTS-guided improvement**. Starting from the supervised checkpoint, roll out negotiations with MCTS and continue training on the best trajectories:

```bash
 python examples/mcts_negotiation_training.py \
  --model_path outputs/craigslist-sft \
  --output_dir outputs/craigslist-mcts \
  --num_samples 128 \
  --rollouts 24
```

Both scripts accept additional flags (see `-h`) to adjust dataset paths, sequence lengths, and search parameters.

3. **Evaluate rollouts**. Compute success rate (SuccessRate), success-within-limits percentage (SL%), and average turns:

```bash
python examples/evaluate_negotiation_metrics.py \
  --input outputs/craigslist-mcts/mcts_dialogues.jsonl \
  --output_dir outputs/evaluation
```

Metrics are saved to `outputs/evaluation/metrics.json`.

### Classic Planning Domains (Non-LLM)

This repository also includes some classic planning domains derived from the original Dyna Gym repo. These examples don't use LLMs but may be useful for debugging purposes.

```bash
python examples/uct_nscartpole_v0.py
python examples/uct_nscartpole_v1.py
...
```
