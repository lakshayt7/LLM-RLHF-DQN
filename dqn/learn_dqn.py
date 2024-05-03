from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, random_agent
import pandas as pd
import ast
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="Train data generator")

parser.add_argument('game', type=str, help='Name of the game to train')
parser.add_argument('training_episodes', type=int, help='Name of traning episodes')
parser.add_argument('generation_episodes', type=int, help='Name of traning episodes')

args = parser.parse_args()

game = args.game
train_episodes = args.training_episodes
generation_episodes = args.generation_episodes

system_prompt = '<s>[INST] <<SYS>> You are a powerful gaming agent who can make proper decisions to beat the user in gaming tasks. You are a helpful assistant that strictly follows the user\'s instructions. <</SYS>>'


if game == 'nim':
    head_prompt = 'In Nim, a strategic game with a set of four piles containing 1, 3, 5, and 7 matches respectively, ' \
            'players aim to avoid taking the last match. During each turn, a player may take any number of matches from a single pile, ' \
            'but must take at least one and cannot exceed the number remaining in that pile. ' \
            'The objective is to force the opponent to pick up the final match, thereby winning the game.' \
            '\n' \
            'The action is presented in [pile:x, take:y], which means take y match(es) from the x-th pile.'
elif game == 'pig':
    head_prompt = 'Pig is a fast-paced dice game where players risk accumulating points with each roll but risk losing them all if they roll a 1. Each player must decide when to stop rolling and bank their points, aiming to be the first to reach 100 points.You are playing Pig with the other.'
elif game == 'liars_dice':
    head_prompt = 'Liar\'s Dice is a game of bluffing and probability, played with two players and each player has 1 dice.' \
           'During each turn, a player can either bid a higher quantity of any particular face value or ' \
           'the same quantity of a higher face value than the previous bid. ' \
           'Each player tries to outbid their opponent without being caught in a lie. ' \
           '\n' \
           'The move in this game is denoted in [x dices, y value], meaning there are at least x dices with face values as y.'
elif game == 'kuhn_poker':
    head_prompt = 'Kuhn poker is a simple model zero-sum two-player imperfect-information game, amenable to a complete game-theoretic analysis. In Kuhn poker, the deck includes only three playing cards: a King (K), a Queen (Q), and a Jack (J).\n' \
           'One card is dealt to each player, and the third is put aside unseen. The players take turns either [Bet] to match the bet raised by the opponent or [Pas] to conceds the game.\n' \
           'If a player bets, the other player must either call the bet by matching it or fold by conceding the game. If both players pass, the game is over, and the player with the higher-ranking card wins. The card rankings are as follows: King (K) > Queen (Q) > Jack (J).\n' \
           '\n' \
           'You are playing Kuhn poker with the opponent. The actions are denoted by [Bet] and [Pass].'
else:
    head_prompt = ''

prompt_name = f"""You must choose an legal action to set up advantages.

Your output must be in the following format:

Action:
Your action wrapped with []

Please return your answer without explanation!
"""

if game == 'nim':

    def get_final_prompt(state):

        piles = state.split(' ')[1:]
        legal_moves = []

        for i, j in enumerate(piles):
            for k in range(1, int(j)+1):
                legal_moves.append((i+1, k))

        legal_move_str = [f"[pile:{move[0]}, take:{move[1]}]" for move in legal_moves]
        legal_move_str = ", ".join(legal_move_str)

        instruction_prompt = f'Currently, the 1st pile has {piles[0]} match(es);\nthe 2nd pile has {piles[1]} match(es);\n' \
                f'the 3rd pile has {piles[2]} match(es);\nthe 4th pile has {piles[3]} match(es). \n\n' \
                f'The legal actions are: {legal_move_str}.'
    
        return f"{system_prompt}\n{head_prompt}\n{prompt_name}\n{instruction_prompt}[/INST]"

elif game == 'pig':
    def get_final_prompt(state):
        txt =state.split(' ')[1:]
    
        legal_move_str = "[roll] [stop]"

        instruction_prompt = f'Right now, your current score is {txt[0]} and your opponent\'s current score is {txt[1][:-1]}. In this turn, you have earned {txt[-1]} score. The legal moves are: {legal_move_str}."'

        return f"{system_prompt}\n{head_prompt}\n{prompt_name}\n{instruction_prompt}[/INST]"

elif game == 'liars_dice':

    def get_final_prompt(last_move, self_dice_face_value, legal_move_str):
        if last_move is None:
            instruction_prompt = f'Currently, the face value of your dice is {self_dice_face_value}. You are the first to go.' \
                    '\n' \
                    'You are playing the Liar\'s Dice with another opponent. Therefore, there are only two dices in total.' \
                    f'\n\n' \
                    f'The legal actions are: {legal_move_str}.'
                    # 'You should call action <Liar> if the opponent called <2 dices, 6 value> in the last round. Because there is no other actions.' \
        else:
            instruction_prompt = f'Currently, the face value of your dice is {self_dice_face_value}. Last time, the opponent called action [{last_move}].' \
                    '\n' \
                    'You are playing the Liar\'s Dice with another opponent. Therefore, there are only two dices in total.' \
                    f'\n\n' \
                    f'The legal actions are: {legal_move_str}.'
                    # 'You should call action <Liar> if the opponent called <2 dices, 6 value> in the last round. Because there is no other actions.' \


        return f"{system_prompt}\n{head_prompt}\n{prompt_name}\n{instruction_prompt}[/INST]"

elif game == 'kuhn_poker':

    def get_final_prompt(id, card, moves):
        print(card)
        print(moves)
        print('id')
        print(id)
        card_mapping = {
            '0': 'Jack (J)',
            '1': 'Queen (Q)',
            '2': 'King (K)'
        }

        player_idx = id

        move_prompt = ''
        if moves is not None:
            move_prompt = 'Here are the past moves in this match:\n'

            for idx, m in enumerate(moves):
                if (player_idx + 1) % (idx + 1) == 0:
                    role = 'you'
                else:
                    role = 'the opponent'

                if m == 'Bet':
                    move = '[Bet]'
                elif m == 'Pass':
                    move = '[Pass]'
                else:
                    raise ValueError

                if idx == 0:
                    move_prompt += f'In the {idx + 1}st round, {role} choose to {move};\n'
                elif idx == 1:
                    move_prompt += f'In the {idx + 1}nd round, {role} choose to {move};\n'
                elif idx == 2:
                    move_prompt += f'In the {idx + 1}rd round, {role} choose to {move};\n'
                else:
                    raise ValueError

        instruction_prompt = f'In this match, your card is {card}.\n' \
                f'{move_prompt}\n' \
                f'Your legal moves are: [Pass], [Bet].'
                

        return f"{system_prompt}\n{head_prompt}\n{prompt_name}\n{instruction_prompt}[/INST]"

storage = pd.DataFrame()
states_str = []
states_raw = []
q_values_l = []
prompts = []

# Create the environment
env = rl_environment.Environment(game)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]

# Create the agents
agents = [
    tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
    for idx in range(num_players)
]

# Train the Q-learning agents in self-play.
for cur_episode in range(train_episodes):
  if cur_episode % 1000 == 0:
    print(f"Episodes: {cur_episode}")
  time_step = env.reset()
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step)
    time_step = env.step([agent_output.action])
  # Episode is over, step all agents with final info state.
  for agent in agents:
    agent.step(time_step)
print("Done!")


# Evaluate the Q-learning agent against a random agent.
from open_spiel.python.algorithms import random_agent
eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]



# Print Q-values and state after training
print("Training completed. Displaying Q-values and states for Agent 0.")
time_step = env.reset()

# Set up a random agent for evaluation

random_agent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)
for cur_episode in range(generation_episodes):
    time_step = env.reset()
    moves = []
    flag = None
    while not time_step.last():
        state_str = str(env.get_state)
        current_state = time_step.observations['info_state'][0]  # Adjust based on how state is stored
        
        # Corrected access to Q-values
        state_key = str(current_state)  # Assuming the state key is a tuple
        q_values = agents[0]._q_values.get(state_key, {})
        
        print(state_key)
        print("\nCurrent State:")
        print(state_str)
        if game == 'pig':
            prompt = get_final_prompt(state_str, moves)
    

        q_values_prc = {}

        for k, v in q_values.items():
            action_str = env.get_state.action_to_string(k)
            q_values_prc[action_str] = v

        if game == 'liars_dice':
            prompt = get_final_prompt(flag, state_str[0],  f"[{'],['.join(list(q_values_prc.keys()))}]")
        if game == 'kuhn_poker':
            if len(state_str) > 0:
                prompt = get_final_prompt(cur_episode%2, state_str[cur_episode%2], moves)
        print(q_values_prc) 
        if q_values:
            states_str.append(state_str)
            states_raw.append(current_state)
            q_values_l.append(q_values_prc)
            prompts.append(prompt)

        player_id = time_step.observations["current_player"]

        if player_id == 0:
            agent_output = agents[player_id].step(time_step, is_evaluation=True)

        else:  # Random agent
           agent_output = random_agent.step(time_step)
            

        action_str = env.get_state.action_to_string(agent_output.action)
        print(action_str)
        moves.append(action_str)
        time_step = env.step([agent_output.action])
        flag = action_str



# Save to DataFrame and CSV
storage['str_state'] = states_str
storage['q_values'] = q_values_l
storage['prompts'] = prompts 
storage.to_csv(f'../data/{game}_q_values.csv', index=False)