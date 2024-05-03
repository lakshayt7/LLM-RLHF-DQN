import pandas as pd
import argparse
import ast 
import numpy as np




def parse_dict(defaultdict_str):
    start_index = defaultdict_str.find('{')
    end_index = defaultdict_str.rfind('}') + 1  # Corrected here: remove the extra +1
    dict_str = defaultdict_str[start_index:end_index]  # Also corrected here
    return ast.literal_eval(dict_str)

def normalize_dict(strdict, m, std_dev):
    dic = parse_dict(strdict)
    normalized_dic = {key: (value - m) / std_dev for key, value in dic.items()}
    return str(normalized_dic)

# Custom action to convert comma-separated input into a list of strings
class CommaSeparatedListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Split the string by commas and strip whitespace
        string_list = [item.strip() for item in values.split(',')]
        setattr(namespace, self.dest, string_list)

parser = argparse.ArgumentParser(description="Merge datasets")

parser.add_argument('--datasets', action=CommaSeparatedListAction, help='A comma-separated list of strings')
parser.add_argument('--base_path', type = str, default="../data/")

args = parser.parse_args()
datasets_dir = [f"{args.base_path}{x}_q_values.csv" for x in args.datasets]

datasets = {}

for name, dir in zip(args.datasets, datasets_dir):
    datasets[name] = pd.read_csv(dir)[['prompts', 'q_values']]
    datasets[name]["game"] = name

    parsed_values = list(datasets[name]["q_values"].apply(lambda x: parse_dict(x).values()))
    flattened_values = [item for sublist in parsed_values for item in sublist]
    mean, std = mean_value = np.mean(flattened_values ), np.std(flattened_values , ddof=1)  

    datasets[name]["q_values"] = datasets[name]["q_values"].apply(lambda x: normalize_dict(x, mean, std))



sampled_dfs = [df.sample(n=min(15000, len(df)), random_state=1) for df in datasets.values()]

# Combine all the sampled DataFrames into one DataFrame
combined_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac = 1)

# Now you can use combined_df as your single DataFrame
combined_df.to_csv(f"../data/{args.base_path}combined_q_values.csv" , index = False)