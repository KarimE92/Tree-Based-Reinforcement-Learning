import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import timeit

env = gym.make('CartPole-v0')

## Creating a Q-Table for each state-action pair
NUM_BUCKETS = (1, 1, 6, 3)  #Take the number of discrete states per observation (to a minimum of 1)
NUM_ACTIONS = env.action_space.n #(take the total number of actions in the environment)
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
#Bounds[1] and Bounds[3] must be set manually as the observation space for those two are unbound
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))




MIN_EXPLORE_RATE = 0.01 #Minimum exploration rate
MIN_LEARNING_RATE = 0.1 #Minimum learning rate


NUM_TRAIN_EPISODES = 200 #Number of training episodes
NUM_TEST_EPISODES = 100 #Number of test episodes
MAX_TRAIN_T = 250 #Maximum training time
MAX_TEST_T = 250 #Maximum testing time
TOTAL_SUCCESS = 50 #The agent must solve the problem this many times consecutively to consider the environment solved

graphx = [] #represents the x coordinate of our graph (episodes)
graphy = [] #represents the y coordinate of our graph (timestep for each episode)
for i in range(0,NUM_TRAIN_EPISODES): #we are going to populate our array so we can plot the x axis properly
    graphx.append(i)


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample() #if we get our explore rate we choose a random action
    else:
        action = np.argmax(q_table[state]) #if we do not get our explore rate we choose the highest QValue from the QTable
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25))) #Explore rate will increase as number of episodes increases

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25))) #Learning rate will increase as number of episodes increases

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)): #we loop through every observation in the observation space
        if state[i] <= STATE_BOUNDS[i][0]: #if the observation is less than the lower bound
            bucket_index = 0 #index is lowest possible value
        elif state[i] >= STATE_BOUNDS[i][1]: #if the observation is greater than the upper bound
            bucket_index = NUM_BUCKETS[i] - 1 #index is highest possible value
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0] #we calculate the difference between the upper and lower bound
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width #We use NUM_BUCKETS to divide the bound_width into segments
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset)) #we see which bound our observation falls in
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


    
def train():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  #the environment does not change over time so our discount factor is very high

    for episode in range(NUM_TRAIN_EPISODES): #For each episode        
        obv, _ = env.reset() #Reset our environment from the last episode
        state_0 = state_to_bucket(obv) #get our current state

        for t in range(MAX_TRAIN_T): #For each timestep
            env.render() #render the environment

            action = select_action(state_0, explore_rate) #select an action
            obv, reward, done, _, _ = env.step(action) #execute action
            state = state_to_bucket(obv) #get our new current state

            # Update the Q based on the result
            best_q = np.amax(q_table[state]) #get the best q of the now current state
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)]) #update the QValue for the previous state

            state_0 = state #prepare for the next timestep by setting our previous state to the current state

            if done: #If the pole has tipped over
               print("Episode %d finished after %f time steps" % (episode, t))
               break #Break out of the current episode to move on to the next episode
        
        graphy.append(t) #Record the time it took for the episode to end for the graph


        # Update parameters for the next episode
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        
        
def test():
    success = 0 #initialize the total number of successes

    for episode in range(NUM_TEST_EPISODES): #for each episode
        obv, _ = env.reset() #reset the environment
        state_0 = state_to_bucket(obv) #get the current state

        tt = 0 #initialise the time taken
        done = False #initialise our loop condition
        while not(done): #while the pole has not tipped over
            tt += 1 #increment time taken
            env.render() #render the environment

            action = select_action(state_0, 0) #select action
            obv, reward, done, _, _ = env.step(action) #execute action
            state_0 = state_to_bucket(obv) #get new state

            if done: #if the pole tips over output the amount of time taken
                print("Test episode %d; time step %f." % (episode, tt))

        if tt >= 200: #if the episode was successful
            success += 1 #increment total successs
            if success >= TOTAL_SUCCESS: #check if environment is solved
                print("Environment Solved!")
                return None #stop training if the environment has been solved
        else:
            success = 0 #reset total successes
            
    print("Environment has not been solved") #If the function does not return None then the environment has not been solved




def graph():
    plt.plot(graphx,graphy) #plot the graph using our 2 arrays for x and y
    plt.xlabel('Episodes') #label x as number of episodes
    plt.ylabel('Time Steps') #label y as timesteps
    plt.title("QLearning Cart Pole Performance") #label the graph
    plt.show() #show the graph
    
if __name__ == "__main__": #This if statement means that if we import this file into another program it will not automatically run the code    
    start = timeit.default_timer() #program start time
    print('Training ...')
    train()
    print("Graphing")
    graph()
    print('Testing ...')
    test()
    stop = timeit.default_timer() #program finish time
    print('Total runtime for QLearning Cart Pole is: ', stop - start, "seconds")
