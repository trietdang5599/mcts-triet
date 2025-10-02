import transformers
from transformers import pipeline

from dyna_gym.pipelines import uct_for_hf_transformer_pipeline

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


# define a reward function based on sentiment of the generated text
sentiment_pipeline = pipeline("sentiment-analysis")
def sentiment_analysis(sentence):
    output = sentiment_pipeline(sentence)[0]
    if output['label'] == 'POSITIVE':
        return output['score']
    else:
        return -output['score']

# maximum number of steps / tokens to generate in each episode
horizon = 50

# arguments for the UCT agent
uct_args = dict(
    rollouts = 20,
    gamma = 1.,
    width = 3,
    alg = 'uct', # or p_uct
)

# will be passed to huggingface model.generate()
model_generation_args = dict(
    top_k = 3,
    top_p = 0.9,
    do_sample = False,
    temperature = 0.7,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
)


pipeline = uct_for_hf_transformer_pipeline(
    model = model,
    tokenizer = tokenizer,
    horizon = horizon,
    reward_func = sentiment_analysis,
    uct_args = uct_args,
    model_generation_args = model_generation_args,
    should_plot_tree = True, # plot the tree after generation
)

input_str = "What do you think about Spider Man movie? \n"
outputs = pipeline(input_str=input_str)

for text, reward in zip(outputs['texts'], outputs['rewards']):
    print("==== Text ====")
    print(text)
    print("==== Reward:", reward, "====")
    print()
