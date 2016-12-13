# Python Version: 3.5.1
__author__ = "Amit Navindgi"
__email__ = "navindgi@usc.edu"
__version__ = "1.0"

import sys
import random
from pprint import pprint
import matplotlib.pyplot as plt

#-----------------GLOBAL VARIABLES SECTION-------------------------

#slots
slots = ["FOODTYPE", "PRICE", "LOCATION"]
#statuses
statuses = ["EMPTY", "FILLED", "CONFIRMED"]
#values each slot can take
food_variables = ['FOODTYPE_EMPTY', 'FOODTYPE_FILLED', 'FOODTYPE_CONFIRMED']
price_variables = ['PRICE_EMPTY', 'PRICE_FILLED', 'PRICE_CONFIRMED']
location_variables = ['LOCATION_EMPTY', 'LOCATION_FILLED', 'LOCATION_CONFIRMED']
#all state variables 3*3=9
state_variables = food_variables + price_variables + location_variables

#system actions
system_actions = ["REQUEST_FOODTYPE", "REQUEST_PRICE", "REQUEST_LOCATION", "CONFIRM_FOODTYPE", "CONFIRM_PRICE", "CONFIRM_LOCATION"]
#user actions
user_actions = ["PROVIDE_FOODTYPE", "PROVIDE_PRICE", "PROVIDE_LOCATION", "YES", "NO"]

#state-action pairs
state_action_pairs = []
#q-values for state-action pairs
q_values = {}
#eligibility traces
e_traces = {}
#rewards
REWARD_METHOD = "STATECHANGE" # change this to "EPISODIC" for alternate reward calculation method
state_rewards = {} #used to reward for every transition to a new state
episodic_reward = 500 #used to calculate reward at end of each episode

#action selection method
ACTION_SELECTION = "E_GREEDY"
#learning method
LEARNING_METHOD = "SARSA" #Q_LEARNING, Q_LAMBDA, SARSA
#learning parameters
random_action_chosen = 0
lambdaq = 0.9
gamma = 0.5 #discount factor
alpha = 0.5 #learning rate
epsilon = 0.2 #exploration probability
epsilon_decrement = 0.005 #to gradually decrease epsilon so as to reduce exploration after every episode

#store status of each slot
TYPE_OF_FOOD = 0 #initially empty
PRICE = 0
LOCATION = 0

#store total reward at end of each episode for final plotting
ALL_REWARDS = []
ALL_EPISODES = []
# REWARDS_QLEARNING = []
# REWARDS_SARSA = []
# REWARDS_QLAMBDA = []
# REWARDS_SARSALAMBDA = []

#to check reward for a interaction
foodcount1 = 0
foodcount2 = 0
pricecount1 = 0
pricecount2 = 0
locationcount1 = 0
locationcount2 = 0

#-----------------END OF GLOBAL VARIABLES SECTION-------------------------

#this function initializes the q-value for all state action pairs to zero
def initialize_q_values():
	global q_values
	global state_action_pairs
	count = 0
	for food_variable in food_variables:
		for price_variable in price_variables:
			for location_variable in location_variables:
				for system_action in system_actions:
					state_action_pairs += [ (food_variable, price_variable, location_variable, system_action) ]
					count += 1
	q_values = {(a, b, c, d) : 0 for a, b, c , d in state_action_pairs}

#this function initializes the eligibility traces for all state-action pairs to zero
def initialize_eligibility_trace():
	global e_traces
	global state_action_pairs
	e_traces = {(a, b, c, d) : 0 for a, b, c , d in state_action_pairs}
	#pprint(len(e_traces))

#this function initializes rewards for each state based on the status of each slot
def initialize_rewards():
	global state_rewards
	for foodtype, price, location, action in state_action_pairs:
		# if((foodtype+price+location).count("EMPTY") == 3):
		# 	state_rewards.update({ (foodtype, price, location, action) : -100 })
		# elif((foodtype+price+location).count("EMPTY") == 2 and (foodtype+price+location).count("FILLED") == 1):
		# 	state_rewards.update({ (foodtype, price, location, action) : 100 })
		# elif((foodtype+price+location).count("EMPTY") == 1 and (foodtype+price+location).count("FILLED") == 2):
		# 	state_rewards.update({ (foodtype, price, location, action) : 200 })
		# elif((foodtype+price+location).count("EMPTY") == 1 and (foodtype+price+location).count("FILLED") == 1 and (foodtype+price+location).count("CONFIRMED") == 1):
		# 	state_rewards.update({ (foodtype, price, location, action) : 300 })
		# elif((foodtype+price+location).count("EMPTY") == 2 and (foodtype+price+location).count("CONFIRMED") == 1):
		# 	state_rewards.update({ (foodtype, price, location, action) : 400 })
		# elif((foodtype+price+location).count("FILLED") == 3):
		# 	state_rewards.update({ (foodtype, price, location, action) : 500 })
		# elif((foodtype+price+location).count("EMPTY") == 1 and (foodtype+price+location).count("CONFIRMED") == 2):
		# 	state_rewards.update({ (foodtype, price, location, action) : 600 })
		# elif((foodtype+price+location).count("FILLED") == 2 and (foodtype+price+location).count("CONFIRMED") == 1):
		# 	state_rewards.update({ (foodtype, price, location, action) : 700 })
		# elif((foodtype+price+location).count("FILLED") == 1 and (foodtype+price+location).count("CONFIRMED") == 2):
		# 	state_rewards.update({ (foodtype, price, location, action) : 800 })
		# else: #((foodtype+price+location).count("CONFIRMED") == 3):
		# 	state_rewards.update({ (foodtype, price, location, action) : 1000 })
		#comment the part below if rewards must be different for every state
		if((foodtype+price+location).count("CONFIRMED") == 3):
		 	state_rewards.update({ (foodtype, price, location, action) : 500 })
		else:
			state_rewards.update({ (foodtype, price, location, action) : 0 })

#this function selects initial state for the system which is when all the slots are empty
def select_initial_state():
	(a, b, c) = ("FOODTYPE_EMPTY", "PRICE_EMPTY", "LOCATION_EMPTY")
	return (a, b, c)

#this function chooses a random action among all possible actions in a given state for exploration
def choose_random_action(current_state):
	food_status, price_status, location_status = current_state
	#get candidate q-value pairs
	possible_actions = { k:v for k,v in q_values.items() if k[0] == food_status and k[1] == price_status and k[2] == location_status }
	chosen_state_action_pair = random.choice(list(possible_actions.items()))
	a, b, c, chosen_action = chosen_state_action_pair[0]
	return chosen_action, chosen_state_action_pair

#this function chooses best action(one with max q-value) among all possible ones in a given state
def choose_best_action(current_state):
	food_status, price_status, location_status = current_state
	#get candidate q-value pairs
	possible_actions = { k:v for k,v in q_values.items() if k[0] == food_status and k[1] == price_status and k[2] == location_status }
	#get maximum q-value for actions from current state
	max_q_value = max(possible_actions.values())
	#get all max q-values in case there are many
	pairs_with_max_q_value = [(key, value) for key, value in possible_actions.items() if value == max_q_value ]
	#choose one randomly
	chosen_state_action_pair = random.choice(pairs_with_max_q_value)
	a, b, c, chosen_action = chosen_state_action_pair[0]
	return chosen_action, chosen_state_action_pair

#this function selects system action based on a selection method, currently only e-greedy method
def select_system_action(current_state):
	global epsilon
	global epsilon_decrement
	if(ACTION_SELECTION == "E_GREEDY"):
		if(random.random() <= epsilon):
			selected_action, selected_state_action_pair = choose_random_action(current_state)
		else:
			selected_action, selected_state_action_pair = choose_best_action(current_state)
		epsilon -= epsilon_decrement
		return selected_action, selected_state_action_pair

#this function selects a user action which is limited by the current systems and status of three slots
def select_user_action(system_action, current_state_action_pair):
	global user_actions
	if(system_action == "REQUEST_FOODTYPE"):
		selected_user_action = "PROVIDE_FOODTYPE"
	elif(system_action == "REQUEST_PRICE"):
		selected_user_action = "PROVIDE_PRICE"
	elif(system_action == "REQUEST_LOCATION"):
		selected_user_action = "PROVIDE_LOCATION"
	elif(system_action == "CONFIRM_FOODTYPE" or system_action == "CONFIRM_PRICE" or system_action == "CONFIRM_LOCATION"):
		selected_user_action = "YES"
	else:
		selected_user_action = "NO"
	return selected_user_action

#this function updates status of the three slots-foodtype, price, location
def update_status(system_action, user_action):
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	if(system_action == "REQUEST_FOODTYPE" and user_action == "PROVIDE_FOODTYPE" and TYPE_OF_FOOD == 0):
		TYPE_OF_FOOD = 1
	elif(system_action == "REQUEST_PRICE" and user_action == "PROVIDE_PRICE" and PRICE == 0):
		PRICE = 1
	elif(system_action == "REQUEST_LOCATION" and user_action == "PROVIDE_LOCATION" and LOCATION ==0):
		LOCATION = 1
	elif(system_action == "CONFIRM_FOODTYPE" and user_action == "YES" and TYPE_OF_FOOD == 1):
		TYPE_OF_FOOD = 2
	elif(system_action == "CONFIRM_PRICE" and user_action == "YES" and PRICE == 1):
		PRICE = 2
	elif(system_action == "CONFIRM_LOCATION" and user_action == "YES" and LOCATION == 1):
		LOCATION = 2
	elif(system_action == "CONFIRM_FOODTYPE" and user_action == "NO" and TYPE_OF_FOOD == 1):
		TYPE_OF_FOOD = 0
	elif(system_action == "CONFIRM_PRICE" and user_action == "NO" and PRICE == 1):
		PRICE = 0
	elif(system_action == "CONFIRM_LOCATION" and user_action == "NO" and LOCATION == 1):
		LOCATION = 0

#this function selects next state based on current status of foodtype, price and location
def select_next_state(current_state_action_pair):
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	#check if food status has changed
	if(TYPE_OF_FOOD == 0):
		food_status = "FOODTYPE_EMPTY"
	elif(TYPE_OF_FOOD == 1):
		food_status = "FOODTYPE_FILLED"
	else:
		food_status = "FOODTYPE_CONFIRMED"
	#check if price status has changed
	if(PRICE == 0):
		price_status = "PRICE_EMPTY"
	elif(PRICE == 1):
		price_status = "PRICE_FILLED"
	else:
		price_status = "PRICE_CONFIRMED"
	#check if location status has changed
	if(LOCATION == 0):
		location_status = "LOCATION_EMPTY"
	elif(LOCATION == 1):
		location_status = "LOCATION_FILLED"
	else:
		location_status = "LOCATION_CONFIRMED"
	return (food_status, price_status, location_status)

#this function calculates reward for one system-user interaction
def calculate_reward_for_this_interaction(reward_for_current_action, system_action, user_action):
	global foodcount1
	global foodcount2
	global pricecount1
	global pricecount2
	global locationcount1
	global locationcount2
	#we have to reward for filling foodtype only the first time since all subsequent 'request and provide foodtype' are rejected
	if(system_action == "REQUEST_FOODTYPE" and user_action == "PROVIDE_FOODTYPE" and TYPE_OF_FOOD == 1 and foodcount1 < 1):
		r = reward_for_current_action - 5
		foodcount1 += 1
	elif(system_action == "REQUEST_PRICE" and user_action == "PROVIDE_PRICE" and PRICE == 1 and pricecount1 < 1):
		r = reward_for_current_action - 5
		pricecount1 += 1
	elif(system_action == "REQUEST_LOCATION" and user_action == "PROVIDE_LOCATION" and LOCATION == 1 and locationcount1 < 1):
		r = reward_for_current_action - 5
		locationcount1 += 1
	elif(system_action == "CONFIRM_FOODTYPE" and user_action == "YES" and TYPE_OF_FOOD == 2 and foodcount2 < 1):
		r = reward_for_current_action - 5
		foodcount2 += 1
	elif(system_action == "CONFIRM_PRICE" and user_action == "YES" and PRICE == 2 and pricecount2 < 1):
		r = reward_for_current_action - 5
		pricecount2 += 1
	elif(system_action == "CONFIRM_LOCATION" and user_action == "YES" and LOCATION == 2 and locationcount2 < 1):
		r = reward_for_current_action - 5
		locationcount2 += 1
	else:
		r = -5
	return r

#this function stores the rewards for each episode for plotting
def store_reward_episode(episode_num, reward_from_this_episode):
	global ALL_EPISODES
	global ALL_REWARDS
	# global REWARDS_QLEARNING
	# global REWARDS_SARSA
	# global REWARDS_QLAMBDA
	# global REWARDS_SARSALAMBDA
	# global LEARNING_METHOD
	ALL_EPISODES.append(episode_num)
	ALL_REWARDS.append(reward_from_this_episode)
	# if LEARNING_METHOD == "Q_LEARNING":
	# 	REWARDS_QLEARNING.append(reward_from_this_episode)
	# elif LEARNING_METHOD == "SARSA":
	# 	REWARDS_SARSA.append(reward_from_this_episode)
	# elif LEARNING_METHOD == "Q_LAMBDA":
	# 	REWARDS_QLAMBDA.append(reward_from_this_episode)
	# elif LEARNING_METHOD == "SARSA_LAMBDA":
	# 	REWARDS_SARSALAMBDA.append(reward_from_this_episode)
	# 	ALL_EPISODES.append(episode_num)
	
#this function plots rewards vs episodes to observe learning rate
def plot_rewards_episodes():
	global ALL_EPISODES
	global ALL_REWARDS
	global LEARNING_METHOD
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title(LEARNING_METHOD)
	plt.plot(ALL_EPISODES, ALL_REWARDS, marker='o', color='r', label='Square')
	print("Rewards for " + str(len(ALL_EPISODES)) + " episodes")
	print(ALL_REWARDS)
	plt.show()

#this function plots combined graph for all learning methods for comparison
def plot_comparison_graph():
	global REWARDS_QLEARNING
	global REWARDS_SARSA
	global REWARDS_QLAMBDA
	global REWARDS_SARSALAMBDA
	global ALL_EPISODES
	print(len(REWARDS_QLEARNING))
	print(len(REWARDS_SARSA))
	print(len(REWARDS_QLAMBDA))
	print(len(REWARDS_SARSALAMBDA))
	print(len(ALL_EPISODES))
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.title('Rewards vs Episodes')
	plt.plot(ALL_EPISODES, REWARDS_QLAMBDA, marker='o', linestyle='--', color='r', label='Square')
	plt.plot(ALL_EPISODES, REWARDS_QLEARNING, marker='o', linestyle='--', color='b', label='Square')
	plt.plot(ALL_EPISODES, REWARDS_SARSA, marker='o', linestyle='--', color='g', label='Square')
	plt.plot(ALL_EPISODES, REWARDS_SARSALAMBDA, marker='o', linestyle='--', color='y', label='Square')
	#print("Rewards for 1000 episodes")
	#print(ALL_REWARDS)
	plt.show()

#this function pretty prints one interaction between system and simulated user
def print_interaction(system_action, user_action, current_state_action_pair, next_state_action_pair):
	global q_values
	print("-------------------------------------------------------")
	print("System : " + system_action)
	print("User   : " + user_action)
	print("Current State-Action Pair : " + str(current_state_action_pair))
	print("Updated Q-value for Current State : " + str(q_values[current_state_action_pair[0]]))
	print("Next State-Action Pair Chosen: " + str(next_state_action_pair))
	
#this function prints values used in the q-learning formula for error checking
def print_values(old , new, reward, max_q_value_next_state):
	print("Old q-value : " + str(old))
	print("Reward : " + str(reward))
	print("Max q-value : " + str(max_q_value_next_state))
	print("New q-value : " + str(new))
	print("-------------------------------------------------------")

#this function starts the interaction between system and simulated user and learns along the way by updating q-values
def start_episode(episode_num):
	total_actions = 0
	total_system_actions = 0
	total_user_actions = 0
	count = 0
	total_reward_from_episode = 0
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	global foodcount1
	global foodcount2
	global pricecount1
	global pricecount2
	global locationcount1
	global locationcount2
	foodcount1 = 0
	foodcount2 = 0
	pricecount1 = 0
	pricecount2 = 0
	locationcount1 = 0
	locationcount2 = 0


	#pprint("reward = " + str(state_rewards[('FOODTYPE_CONFIRMED', 'PRICE_EMPTY', 'LOCATION_EMPTY', 'CONFIRM_LOCATION')]))
	current_state = select_initial_state() #("FOODTYPE_EMPTY", "PRICE_EMPTY", "LOCATION_EMPTY")

	while(total_actions < 40 and (TYPE_OF_FOOD < 2 or PRICE < 2 or LOCATION < 2)):
		#select a system action using a selection method like e-greedy
		system_action, current_state_action_pair = select_system_action(current_state)
		#select a corresponding user action
		user_action = select_user_action(system_action, current_state_action_pair)
		#update status of slots and go to next state only if user action is a valid one
		update_status(system_action, user_action)
		#get the next state represented in form of a tuple of statuses of three slots
		next_state = select_next_state(current_state_action_pair)
		#get the next state-action pair with max q-value
		next_action, next_state_action_pair = choose_best_action(next_state)
		#check reward for current action being taken i.e reward for going to next state
		reward_for_current_action = state_rewards[next_state_action_pair[0]]
		#store old q-value of current state-action pair for logs
		old_q_value = q_values[current_state_action_pair[0]]
		#calculate interaction and episode rewards
		total_reward_from_interaction = calculate_reward_for_this_interaction(reward_for_current_action, system_action, user_action)
		total_reward_from_episode += total_reward_from_interaction
		#update q-value of current state-action pair
		q_values[current_state_action_pair[0]] = q_values[current_state_action_pair[0]] + alpha * ( total_reward_from_interaction + (gamma * q_values[next_state_action_pair[0]]) - q_values[current_state_action_pair[0]])
		#print a log this interaction
		#print_interaction(system_action, user_action, current_state_action_pair, next_state_action_pair)
		#print_values(old_q_value, q_values[current_state_action_pair[0]], total_reward_from_interaction, q_values[next_state_action_pair[0]])
		#set next state and current state and continue until terminal state is reached
		current_state = next_state
		count += 1

	print(str(count) + " interactions in episode " + str(episode_num))
	#reset status for new episode
	TYPE_OF_FOOD = 0
	PRICE = 0
	LOCATION = 0
	return total_reward_from_episode

#this function chooses an action for next state using e-greedy technique
def choose_next_action_qlambda(current_state):
	global epsilon
	global epsilon_decrement
	global random_action_chosen
	if(ACTION_SELECTION == "E_GREEDY"):
		if(random.random() <= epsilon):
			random_action_chosen = 1
			selected_action, selected_state_action_pair = choose_random_action(current_state)
		else:
			random_action_chosen = 0
			selected_action, selected_state_action_pair = choose_best_action(current_state)
		epsilon -= epsilon_decrement
		return selected_action, selected_state_action_pair

#this function starts Watkin's Q(lambda) algorithm
def start_q_lambda(episode_num):
	total_actions = 0
	total_system_actions = 0
	total_user_actions = 0
	count = 0
	total_reward_from_episode = 0
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	global foodcount1
	global foodcount2
	global pricecount1
	global pricecount2
	global locationcount1
	global locationcount2
	foodcount1 = 0
	foodcount2 = 0
	pricecount1 = 0
	pricecount2 = 0
	locationcount1 = 0
	locationcount2 = 0

	#initialize e values to 0 for every episode
	initialize_eligibility_trace()
	#initialize state
	current_state = select_initial_state() #("FOODTYPE_EMPTY", "PRICE_EMPTY", "LOCATION_EMPTY")
	#select a system action using a selection method like e-greedy
	system_action, current_state_action_pair = select_system_action(current_state)
		
	
	while(total_actions < 40 and (TYPE_OF_FOOD < 2 or PRICE < 2 or LOCATION < 2)):
		#select a corresponding user action
		user_action = select_user_action(system_action, current_state_action_pair)
		#update status of slots and go to next state only if user action is a valid one
		update_status(system_action, user_action)
		#get the next state represented in form of a tuple of statuses of three slots
		next_state = select_next_state(current_state_action_pair)
		#get the next state-action pair with max q-value
		next_action, next_state_action_pair = choose_next_action_qlambda(next_state)
		#check reward for current action being taken i.e reward for going to next state
		reward_for_current_action = state_rewards[next_state_action_pair[0]]
		#calculate interaction and episode rewards
		total_reward_from_interaction = calculate_reward_for_this_interaction(reward_for_current_action, system_action, user_action)
		total_reward_from_episode += total_reward_from_interaction
		#check if action chosen was random or not and update A* accordingly
		if(random_action_chosen == 0):
			best_action = next_action #action A* 
			best_state_action_pair = next_state_action_pair
			#print(best_state_action_pair)
		else:
			best_action, best_state_action_pair = choose_best_action(next_state) #action with max q value
		#calculate delta
		delta = total_reward_from_interaction + (gamma * q_values[best_state_action_pair[0]]) - q_values[current_state_action_pair[0]]
		#increment z
		e_traces[current_state_action_pair[0]] += 1
		#update q values for all sa pairs
		for k,v in q_values.items():
			q_values[k] = q_values[k] + alpha * delta * e_traces[k]
			#pprint(q_values[k])
			if(random_action_chosen == 0):
				e_traces[k] = gamma * lambdaq * e_traces[k]
			else:
				e_traces[k] = 0
		#update state S to S' and action A to A'
		system_action = next_action
		current_state_action_pair = next_state_action_pair

		count += 1

	print(str(count) + " interactions in episode " + str(episode_num))
	#reset status of slots
	TYPE_OF_FOOD = 0
	PRICE = 0
	LOCATION = 0

	return total_reward_from_episode

#this function starts sarsa on-policy learning method
def start_sarsa(episode_num):
	total_actions = 0
	total_system_actions = 0
	total_user_actions = 0
	count = 0
	total_reward_from_episode = 0
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	global foodcount1
	global foodcount2
	global pricecount1
	global pricecount2
	global locationcount1
	global locationcount2
	foodcount1 = 0
	foodcount2 = 0
	pricecount1 = 0
	pricecount2 = 0
	locationcount1 = 0
	locationcount2 = 0

	#initialize S
	current_state = select_initial_state() #("FOODTYPE_EMPTY", "PRICE_EMPTY", "LOCATION_EMPTY")
	#select a system action using a selection method like e-greedy
	system_action, current_state_action_pair = select_system_action(current_state)
		
	while(total_actions < 40 and (TYPE_OF_FOOD < 2 or PRICE < 2 or LOCATION < 2)):
		#select a corresponding user action
		user_action = select_user_action(system_action, current_state_action_pair)
		#update status of slots and go to next state only if user action is a valid one
		update_status(system_action, user_action)
		#get the next state represented in form of a tuple of statuses of three slots
		next_state = select_next_state(current_state_action_pair)
		#get the next state-action pair with max q-value
		next_action, next_state_action_pair = select_system_action(next_state)
		#check reward for current action being taken i.e reward for going to next state
		reward_for_current_action = state_rewards[next_state_action_pair[0]]
		#store old q-value of current state-action pair for logs
		old_q_value = q_values[current_state_action_pair[0]]
		#calculate interaction and episode rewards
		total_reward_from_interaction = calculate_reward_for_this_interaction(reward_for_current_action, system_action, user_action)
		total_reward_from_episode += total_reward_from_interaction
		#update q-value of current state-action pair
		q_values[current_state_action_pair[0]] = q_values[current_state_action_pair[0]] + alpha * ( total_reward_from_interaction + (gamma * q_values[next_state_action_pair[0]]) - q_values[current_state_action_pair[0]])
		#print a log this interaction
		#print_interaction(system_action, user_action, current_state_action_pair, next_state_action_pair)
		#print_values(old_q_value, q_values[current_state_action_pair[0]], total_reward_from_interaction, q_values[next_state_action_pair[0]])
		#set next state and current state and continue until terminal state is reached
		current_state = next_state
		system_action = next_action
		current_state_action_pair = next_state_action_pair
		count += 1

	print(str(count) + " interactions in episode " + str(episode_num))
	#reset status for new episode
	TYPE_OF_FOOD = 0
	PRICE = 0
	LOCATION = 0
	return total_reward_from_episode

#this function starts sarsa with eligibility traces learning method
def start_sarsa_lambda(episode_num):
	total_actions = 0
	total_system_actions = 0
	total_user_actions = 0
	count = 0
	total_reward_from_episode = 0
	global TYPE_OF_FOOD
	global PRICE
	global LOCATION
	global foodcount1
	global foodcount2
	global pricecount1
	global pricecount2
	global locationcount1
	global locationcount2
	foodcount1 = 0
	foodcount2 = 0
	pricecount1 = 0
	pricecount2 = 0
	locationcount1 = 0
	locationcount2 = 0

	#initialize e values to 0 for every episode
	initialize_eligibility_trace()
	#initialize state
	current_state = select_initial_state() #("FOODTYPE_EMPTY", "PRICE_EMPTY", "LOCATION_EMPTY")
	#select a system action using a selection method like e-greedy
	system_action, current_state_action_pair = select_system_action(current_state)
		
	
	while(total_actions < 40 and (TYPE_OF_FOOD < 2 or PRICE < 2 or LOCATION < 2)):
		#select a corresponding user action
		user_action = select_user_action(system_action, current_state_action_pair)
		#update status of slots and go to next state only if user action is a valid one
		update_status(system_action, user_action)
		#get the next state represented in form of a tuple of statuses of three slots
		next_state = select_next_state(current_state_action_pair)
		#get the next state-action pair with max q-value
		next_action, next_state_action_pair = choose_next_action_qlambda(next_state)
		#check reward for current action being taken i.e reward for going to next state
		reward_for_current_action = state_rewards[next_state_action_pair[0]]
		#calculate interaction and episode rewards
		total_reward_from_interaction = calculate_reward_for_this_interaction(reward_for_current_action, system_action, user_action)
		total_reward_from_episode += total_reward_from_interaction
		#calculate delta
		delta = total_reward_from_interaction + (gamma * q_values[next_state_action_pair[0]]) - q_values[current_state_action_pair[0]]
		#increment z
		e_traces[current_state_action_pair[0]] += 1
		#update q values for all sa pairs
		for k,v in q_values.items():
			q_values[k] = q_values[k] + alpha * delta * e_traces[k]
			#pprint(q_values[k])
			e_traces[k] = gamma * lambdaq * e_traces[k]
			
		#update state S to S' and action A to A'
		system_action = next_action
		current_state_action_pair = next_state_action_pair

		count += 1

	print(str(count) + " interactions in episode " + str(episode_num))
	#reset status of slots
	TYPE_OF_FOOD = 0
	PRICE = 0
	LOCATION = 0

	return total_reward_from_episode

if __name__ == '__main__':
	initialize_q_values()
	initialize_rewards()

	#to change the learning method, please change the value of LEARNING_METHOD in global variables section
	#LEARNING_METHOD = "Q_LEARNING"
	if LEARNING_METHOD == "Q_LEARNING":
		for episode_num in range(0,50):
			reward_from_this_episode = start_episode(episode_num)
			store_reward_episode(episode_num, reward_from_this_episode)
		plot_rewards_episodes()
	#LEARNING_METHOD = "Q_LAMBDA"
	if LEARNING_METHOD == "Q_LAMBDA":
		for episode_num in range(0,50):
			reward_from_this_episode = start_q_lambda(episode_num)
			store_reward_episode(episode_num, reward_from_this_episode)
		plot_rewards_episodes()
	#LEARNING_METHOD = "SARSA"
	if LEARNING_METHOD == "SARSA":
		for episode_num in range(0,50):
			reward_from_this_episode = start_sarsa(episode_num)
			store_reward_episode(episode_num, reward_from_this_episode)
		plot_rewards_episodes()
	#LEARNING_METHOD = "SARSA_LAMBDA"
	if LEARNING_METHOD == "SARSA_LAMBDA":
		for episode_num in range(0,50):
			reward_from_this_episode = start_sarsa_lambda(episode_num)
			store_reward_episode(episode_num, reward_from_this_episode)
		plot_rewards_episodes()
	#plot_comparison_graph()