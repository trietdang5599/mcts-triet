"""MCTS-guided negotiation fine-tuning pipeline for Craigslist Bargains."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.utils import is_accelerate_available

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dyna_gym.data_utils import (
    CRAIGSLIST_SPECIAL_TOKENS,
    load_craigslist_split,
    render_dialogue,
)
from dyna_gym.data_utils.craigslist import (
    NegotiationExample,
    extract_final_price,
)
from dyna_gym.evaluation.metrics import NegotiationRecord, compute_negotiation_metrics
from dyna_gym.pipelines import uct_for_hf_transformer_pipeline


try:
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import GradientAccumulationPlugin
except Exception:  # pragma: no cover - accelerate optional
    Accelerator = None
    DataLoaderConfiguration = None
    GradientAccumulationPlugin = None

if Accelerator is not None and DataLoaderConfiguration is not None:
    def _patched_create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = (
            GradientAccumulationPlugin(**grad_acc_kwargs) if GradientAccumulationPlugin else None
        )

        accelerator_kwargs = dict(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )
        accelerator_kwargs["dataloader_config"] = DataLoaderConfiguration(
            dispatch_batches=self.args.dispatch_batches,
            split_batches=self.args.split_batches,
        )

        self.accelerator = Accelerator(**accelerator_kwargs)
        self.gather_function = self.accelerator.gather_for_metrics
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    Trainer.create_accelerator_and_postprocess = _patched_create_accelerator_and_postprocess


def build_prompt(example: NegotiationExample, prompt_turns: int) -> str:
    return render_dialogue(example, include_outcome=False, max_turns=prompt_turns)


def make_reward_fn(example: NegotiationExample):
    buyer_target = example.buyer_target or example.list_price or 0.0
    seller_target = example.seller_target or example.list_price or buyer_target
    list_price = example.list_price or seller_target or buyer_target or 1.0

    def reward_fn(text: str) -> float:
        lowered = text.lower()
        deal_mentions = lowered.count("deal reached")
        if deal_mentions != 1:
            return -1.0

        price = extract_final_price(text)
        if price is None:
            return -1.0

        price_token = f"${price:,.2f}"
        if text.count(price_token) != 1:
            return -1.0

        within_interval = 1.0 if buyer_target <= price <= seller_target else 0.0
        proximity = 1.0 - min(abs(price - list_price) / max(list_price, 1.0), 1.0)

        length_penalty = 0.0
        turns = lowered.count("buyer") + lowered.count("seller")
        if turns > 16:
            length_penalty = 0.05 * (turns - 16)

        return 0.6 * within_interval + 0.4 * proximity - length_penalty

    return reward_fn


def tokenize_texts(tokenizer, dataset: Dataset, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=['text'])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=Path('dataset/craigslist_bargains'))
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the initial fine-tuned model checkpoint',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/craigslist-mcts'),
    )
    parser.add_argument('--num_samples', type=int, default=128, help='Number of dialogues to generate with MCTS')
    parser.add_argument(
        '--prompt_turns',
        type=int,
        default=2,
        help='Number of ground-truth turns to condition on',
    )
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=16)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--rollouts', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--reuse_tree', action='store_true', help='Reuse MCTS tree between steps')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='Only generate dialogues, skip further fine-tuning',
    )
    parser.add_argument(
        '--should_print_tree',
        action='store_true',
        help='Print search trees during generation',
    )
    parser.add_argument(
        '--eval_samples',
        type=int,
        default=None,
        help='Number of test dialogues to evaluate; use entire test split by default.',
    )
    return parser.parse_args()


def build_generator(model, tokenizer, args, *, should_print_tree: Optional[bool] = None):
    if should_print_tree is None:
        should_print_tree = args.should_print_tree

    generation_args = dict(
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    uct_args = dict(
        rollouts=args.rollouts,
        gamma=1.0,
        width=args.top_k,
        reuse_tree=args.reuse_tree,
    )
    return uct_for_hf_transformer_pipeline(
        model=model,
        tokenizer=tokenizer,
        horizon=args.horizon,
        reward_func=lambda _: 0.0,
        uct_args=uct_args,
        model_generation_args=generation_args,
        should_plot_tree=False,
        should_print_tree=should_print_tree,
        reward_func_input_is_state=False,
        decode_skip_special_tokens=False,
    )


def generate_dialogues(
    examples: Iterable[NegotiationExample],
    generator,
    tokenizer,
    model,
    prompt_turns: int,
) -> List[dict]:
    dialogues: List[dict] = []
    model.eval()
    for example in examples:
        prompt = build_prompt(example, prompt_turns)
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0).to(model.device)
        attention_mask = inputs['attention_mask'].squeeze(0).to(model.device)

        reward_fn = make_reward_fn(example)
        with torch.no_grad():
            outputs = generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                reward_override=reward_fn,
                skip_special_tokens=False,
            )
        texts = outputs.get('texts')
        if not texts:
            continue

        rewards = outputs['rewards']
        best_idx = max(range(len(rewards)), key=lambda idx: rewards[idx])
        text_with_tokens = outputs['texts_with_special_tokens'][best_idx]
        plain_text = outputs['texts_plain'][best_idx]
        reward = float(rewards[best_idx])
        turn_count = text_with_tokens.count('<buyer>') + text_with_tokens.count('<seller>')

        dialogues.append(
            {
                'text': text_with_tokens,
                'plain_text': plain_text,
                'reward': reward,
                'scenario_id': example.scenario_id,
                'buyer_target': example.buyer_target,
                'seller_target': example.seller_target,
                'list_price': example.list_price,
                'category': example.category,
                'title': example.title,
                'turn_count': turn_count,
            }
        )
    return dialogues


def save_dialogues_jsonl(path: Path, dialogues: List[dict]) -> None:
    with path.open('w', encoding='utf-8') as fh:
        for payload in dialogues:
            fh.write(json.dumps(payload) + '\n')


def dialogues_to_records(dialogues: Iterable[dict]) -> List[NegotiationRecord]:
    records: List[NegotiationRecord] = []
    for dialogue in dialogues:
        records.append(
            NegotiationRecord(
                text=dialogue.get('text', ''),
                plain_text=dialogue.get('plain_text'),
                reward=float(dialogue.get('reward', 0.0)),
                buyer_target=dialogue.get('buyer_target'),
                seller_target=dialogue.get('seller_target'),
                turn_count=dialogue.get('turn_count'),
            )
        )
    return records


def serialise_mapping(values: dict) -> dict:
    serialised = {}
    for key, value in values.items():
        if isinstance(value, numbers.Number):
            serialised[key] = float(value)
        else:
            serialised[key] = value
    return serialised


def evaluate_on_test_split(
    args: argparse.Namespace,
    model,
    tokenizer,
    trainer: Trainer,
) -> None:
    test_examples = load_craigslist_split(args.data_dir, 'test')
    if args.eval_samples is not None and args.eval_samples < len(test_examples):
        random.seed(args.seed)
        test_examples = random.sample(test_examples, args.eval_samples)

    if not test_examples:
        print('No test examples found; skipping evaluation.')
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    eval_generator = build_generator(model, tokenizer, args, should_print_tree=False)
    test_dialogues = generate_dialogues(
        test_examples,
        eval_generator,
        tokenizer,
        model,
        args.prompt_turns,
    )
    test_dialogues_path = args.output_dir / 'test_generated_dialogues.jsonl'
    save_dialogues_jsonl(test_dialogues_path, test_dialogues)

    negotiation_metrics = compute_negotiation_metrics(dialogues_to_records(test_dialogues))
    negotiation_metrics_path = args.output_dir / 'test_negotiation_metrics.json'
    with negotiation_metrics_path.open('w', encoding='utf-8') as fh:
        json.dump(serialise_mapping(negotiation_metrics), fh, indent=2)

    print('Negotiation metrics on test split:')
    print(json.dumps(negotiation_metrics, indent=2))

    test_dataset = Dataset.from_dict(
        {'text': [render_dialogue(example) for example in test_examples]}
    )
    tokenized_test = tokenize_texts(tokenizer, test_dataset, args.max_length)
    lm_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix='test')
    if 'test_loss' in lm_metrics:
        try:
            lm_metrics['test_perplexity'] = math.exp(lm_metrics['test_loss'])
        except OverflowError:
            lm_metrics['test_perplexity'] = float('inf')

    lm_metrics_path = args.output_dir / 'test_language_model_metrics.json'
    with lm_metrics_path.open('w', encoding='utf-8') as fh:
        json.dump(serialise_mapping(lm_metrics), fh, indent=2)

    print('Language modeling metrics on test split:')
    print(json.dumps(lm_metrics, indent=2))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens(CRAIGSLIST_SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))

    train_generator = build_generator(model, tokenizer, args)

    train_examples = load_craigslist_split(args.data_dir, 'train')
    random.shuffle(train_examples)
    if args.num_samples and args.num_samples < len(train_examples):
        train_examples = train_examples[: args.num_samples]

    generated_dialogues = generate_dialogues(
        train_examples,
        train_generator,
        tokenizer,
        model,
        args.prompt_turns,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dialogues_path = args.output_dir / 'mcts_dialogues.jsonl'
    save_dialogues_jsonl(train_dialogues_path, generated_dialogues)
    print(f'Saved {len(generated_dialogues)} generated dialogues to {train_dialogues_path}')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized_train = None
    if generated_dialogues:
        dataset = Dataset.from_dict({'text': [dialogue['text'] for dialogue in generated_dialogues]})
        tokenized_train = tokenize_texts(tokenizer, dataset, args.max_length)
    else:
        print('No dialogues generated; skipping fine-tuning step.')

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy='no',
        save_total_limit=args.save_total_limit,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if not args.no_train and tokenized_train is not None:
        trainer.train()
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(args.output_dir)

    model = trainer.model
    model.eval()

    evaluate_on_test_split(args, model, tokenizer, trainer)


if __name__ == '__main__':
    main()
