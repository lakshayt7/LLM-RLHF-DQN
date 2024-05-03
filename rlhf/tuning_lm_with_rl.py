
import os
import pandas as pd

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset

from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    pipeline
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftConfig,
    PeftModel
)

import ast
import re
from torch.nn.utils.rnn import pad_sequence

import bitsandbytes as bnb


from transformers import BitsAndBytesConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

regexes = {
    "tic_tac_toe": 'C[1-3]R[1-3]',
    "connect4": '(C[1-7]|Column.{0,3}[1-7]|column.{0,3}[1-7])',
    'nim' : '(pile:1, take:1|pile:2, take:[1-3]|pile:3, take:[1-5]|pile:4, take:[1-7])',
    'kuhn_poker' : "(Pass|Bet)",
    'pig' : '(roll|stop|Roll|Stop)', 
    'liars_dice': '([1-2]-[1-6]|Liar)', 
    'first_sealed_auction': '(?:[0-9]|10)' 
}

import os, sys
os.environ['TRANSFORMERS_CACHE'] = '/gpfs/scratch/lt2504/.cache/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/gpfs/scratch/lt2504/.cache/huggingface/datasets'

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


load_in_4bit =  True
llm_int8_threshold =  6.0
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype =  "bfloat16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    llm_int8_threshold=llm_int8_threshold,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
)


tqdm.pandas()

def parse_dict(defaultdict_str):
    start_index = defaultdict_str.find('{')
    end_index = defaultdict_str.rfind('}') + 1  # Corrected here: remove the extra +1
    dict_str = defaultdict_str[start_index:end_index]  # Also corrected here
    return ast.literal_eval(dict_str)

def extract_numbers(input_string):
    # Pattern to match numbers before 'dice' and 'value'
    pattern = r'(\d+)\s*dice,\s*(\d+)\s*value'
    
    # Search the pattern in the input string
    match = re.search(pattern, input_string)
    
    # Check if a match is found
    if match:
        # Extract and convert the numbers
        num_dice = int(match.group(1))
        num_value = int(match.group(2))
        return num_dice, num_value
    else:
        # Return a default or an error if no match is found
        return None

def get_value(response, v_dict, game):
    # Regex pattern to find the required format
    pattern = regexes[game]

    # Find all occurrences of the pattern
    matches = re.findall(pattern, response, re.IGNORECASE)


    if len(matches) == 0:
        return torch.tensor(-1, dtype=torch.float32)

    if game == 'tic_tac_toe':
        print((int(matches[0][1]) - 1)*3 + int(matches[0][3]) - 1)
        try:
            pos = (int(matches[0][1]) - 1)*3 + int(matches[0][3]) - 1
        except:
            return torch.tensor(-1, dtype=torch.float32)
    elif game == 'nim':
         pos = f"{matches[0]};"
    elif game == 'kuhn_poker':
        pos = matches[0]
    elif game == 'first_sealed_auction':
        pos = matches[0]
    elif game == 'liars_dice':
        pos = matches[0]
    elif game == 'pig':
        pos = matches[0].lower()
      
    else:
        pos = -1
    return torch.tensor(v_dict.get(pos, -1), dtype=torch.float32)

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=456, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)



def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    train_dataset = load_dataset(
        "csv",
        data_files=dataset_name,
        delimiter=",",  # Change this if your delimiter is different
        quotechar='"',  # Change or remove if your quoting is different
        split="train",
        cache_dir=None
    )
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "re": [],
            "game": []
        }
        #print(examples.keys())
        n = len(examples["prompts"])
        for i in range(n):
            question = examples["prompts"][i]
            reward = examples["q_values"][i]
            game = examples["game"][i]
            '''
            tokenized_question = tokenizer(question, padding="max_length",  # Pad all samples to max length
                                truncation=True,       # Truncate to max model input length
                                max_length=400  # Ensure all tokens are below the model max length)
                                
            )
            '''
            tokenized_question = tokenizer(question)
            new_examples["query"].append(question)
            new_examples["input_ids"].append(tokenized_question["input_ids"])
            new_examples["re"].append(reward)
            new_examples["game"].append(game)
        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])




config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)


tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name,
                                              add_bos_token=True,
                                              add_eos_token=True, 
                                              padding_side='left')
tokenizer.add_special_tokens({'pad_token':"[PAD]"})
'''

tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name, padding_side="right", add_eos_token=True)
# required for llama

'''


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)


# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

#quantization_config = BitsAndBytesConfig(bits=8)  # Specify the quantization bits as per your requirements

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    config.model_name, load_in_8bit=True, device_map={"": Accelerator().process_index}
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.resize_token_embeddings(len(tokenizer)) 

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model,
    quantization_config=bnb_config,
    device_map={"": current_device},
    peft_config=lora_config
)


optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)


# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_length": 512,  # You can set this based on the maximum expected length of the output
    "max_new_tokens": 400  # Alternatively, set max_new_tokens instead of max_length
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)
 
ctr = 0


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    ctr +=1
    question_tensors = batch['input_ids']#.to(device)

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )

    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    responses = batch["response"]
    rewards_dict = batch["re"]
    games = batch["game"]

    
    
    n = len(responses)
    rewards = [get_value(responses[i], parse_dict(rewards_dict[i]), games[i])  for i in range(n)]

    #rewards = [torch.tensor(1.0)]*len(responses)
    # Run PPO steps
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

    try:
        ppo_trainer.log_stats(stats, batch, rewards)
    except:
        print(stats)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}") 
    

print(f'saved at {script_args.output_dir }')
ppo_trainer.save_pretrained(script_args.output_dir + f"_final")   