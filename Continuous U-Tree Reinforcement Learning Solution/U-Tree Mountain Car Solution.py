import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.stats import kstest
import timeit

#Defining our contants
NUM_SPLIT_EPISODES = 150 #Number of splitting episodes
NUM_TRAIN_EPISODES = 200 #Number of training episodes
NUM_TEST_EPISODES = 5 #Number of testing episodes
MAX_TRAIN_TIME = 250 #Maximum training time
TOTAL_SUCCESS = 50 #Total number of consecutive successes for the environment to be considered solved

MIN_EXPLORE_RATE = 0.1 #Minimum Exploration Rate
MIN_LEARNING_RATE = 0.1 #Minimum Learning Rate
discount_factor = 0.99 #Environment does not change over time so discount factor is high

STOPPING_CRITERIA = 0.0000000000000000000000000000000001 #Since values are very small we must have a small stopping criteria for our splits

env = gym.make("MountainCar-v0")


graphx = []
graphy = []
for i in range(0,NUM_TRAIN_EPISODES):
    graphx.append(i)


def select_action(currentnode, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()#if we get our explore rate we choose a random action
    else:
        action = np.argmax(currentnode.GetQ()) #if we do not get our explore rate we choose the highest QValue from the QTable
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25))) #Explore rate will increase as number of episodes increases

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25))) #Learning rate will increase as number of episodes increases

 
class Node:
    def __init__(self):
        #When we initilise a node it does not point to any other node so left and right are None
        self.left = None 
        self.right = None
        self.obv = [None] * len(env.observation_space.sample()) #This will be an array of observations where they will all be None except for the observation we are using to move down the decision tree
        self.obvtuple = [] # [previous obv, action, current obv, reward, expected reward value]

        self.Q = [0] * (env.action_space.n) #this will be an array of QValues with len(Q) = len(Action_Space) and will be what we use to decide what action to take

        self.split = 0 #the number of splits that have occurred in the decision tree as a result of the algorithm
        self.splitindex = [0] * len(env.observation_space.sample()) #a count of which observations were split
        
    def GetState(self, observation):
        if self.left is not None and self.right is not None: #if we are not at a leaf node
            for i in range(0, len(self.obv)): #loop through the observation space
                if self.obv[i] is not None: #if the observation is the decision observation
                    if observation[i] < self.obv[i]: 
                        return self.left.GetState(observation) #if our observation is less than the decision's we move left on the tree
                    elif observation[i] >= self.obv[i]:
                        return self.right.GetState(observation) #else we move right
        else:
            return self #when we reach the leaf node we just return that leaf

    def incsplit(self, index): #when a split occurs we increment the number 
        self.split+=1
        self.splitindex[index]+=1
        
    def getsplit(self): #getter method for split
        return self.split
    
    def getsplitindex(self): #getter method for splitindex
        return self.splitindex
    
    def SetObv(self, index, obv): #we use this to set the observation we use to make decisions for moving down the decision tree
        self.obv[index] = obv
    
    def GetQ(self): #getter method for our QValues
        return self.Q

    def SetQ(self, obvtuple, learning_rate): #setter method for our QValues
        best_q = np.amax(self.Q)
        self.Q[obvtuple[1]] += learning_rate*(obvtuple[3] + discount_factor*(best_q) - (root.GetState(obvtuple[0]).GetQ()[obvtuple[1]]))
                
    def AddObvTuple(self, obvtuple):
        self.obvtuple.append(obvtuple)

        
    def Processing(self):
        #We need to get to the leaf nodes in order to perform processing, so we call the processing function recursively until we reach the leaf nodes
        if self.left is not None and self.right is not None:
            self.left.Processing()
            self.right.Processing()
            return None #we return none here because we do not want to perform processing on non-leaf nodes

        #here we initialize values so we can determine what the best split for the leaf is
        bestkolsmitest = [0, 0]
        bestkolsmiobv = []
        bestkolsmiobvindex = 0
        bestfirsthalf = []
        bestsecondhalf = []
        bestkolsmiflag = False

        #now we need to calculate the QValue of each transition tuple
        for index in range(0, len(self.obvtuple)):
            self.obvtuple[index][4] = self.obvtuple[index][3] + discount_factor * max(root.GetState(self.obvtuple[index][2]).GetQ()) #The issue with this equation is since all the rewards are the same, all the values are the same, and so the distribution of data is the same as well, so we cannot use this for the kolmogorov smirnov test unless we write our own reward function
            
        #we loop through every observation so we can create a split in the decision tree
        for obvindex in range(0, len(env.observation_space.sample())): #for each observation
            sortedtuple = sorted(self.obvtuple, key=lambda observations: observations[0][obvindex]) #Sorting by current observation in the tuple by i (ascending)
            #we need to split the data using a loop
            for splitindex in range(0, len(sortedtuple)): #for each possible split
                #we seperate the observation tuples based on our split                
                first_half = sortedtuple[:splitindex]
                second_half = sortedtuple[splitindex:]

                #now that we have the split we need to populate new arrays so we can plot the cumulative distribution of the split we are trying to test
                data1 = []
                data2 = []
                for temp in range(0,splitindex):
                    data1.append(sortedtuple[temp][4]) #append the qvalues to data1
                for temp in range(splitindex, len(sortedtuple)):
                    data2.append(sortedtuple[temp][4]) #append the qvalues to data2

                #We now perform Kolmogorov-Smirnov tests on the two new datasets to find how good the split between the two datasets is
                if data1 != [] and data2 != []:
                    kolsmitest = kstest(data1, data2) #Ktest is an array of [statistic, pvalue]


                    
                    if kolsmitest[0] > bestkolsmitest[0] and kolsmitest[1] < STOPPING_CRITERIA: #if the new split is better than our previous best split AND satisfies our stopping criteria (meaning it is statistically significant)
                        #log all information related to the split
                        bestkolsmitest = kolsmitest
                        bestkolsmiobv = self.obvtuple[splitindex][2]
                        bestkolsmiobvindex = obvindex
                        bestfirsthalf = first_half
                        bestsecondhalf = second_half
                        bestkolsmiflag = True

        if bestkolsmiflag: #if a valid, best split was found
            root.incsplit(bestkolsmiobvindex) #increment the number of splits in the decision tree

            self.left = Node() #create a new left node
            self.right = Node() #create a new right node

            self.SetObv(bestkolsmiobvindex, bestkolsmiobv[bestkolsmiobvindex]) #set our decision for moving down the decision tree

            #Now that we have created the split we need to erase the QValues in the current node as well as the obvtuples stored in that node (that has now become a decision)
            self.Q = [0] * (env.action_space.n)
            self.obvtuple = []               


 
def split():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    for episodes in range(1,NUM_SPLIT_EPISODES+1): #for each episode
        obv, _ = env.reset() #reset the environment
        for timestep in range(0, MAX_TRAIN_TIME):#for each timestep
            env.render() #render the environment

            currentnode = root.GetState(obv) #get the current state
            action = select_action(currentnode, explore_rate) #select an action
            prevobv = obv #log the previous observation
            obv, reward, done, _, _ = env.step(action) #execute the action
            currentnode = root.GetState(obv) #get the new state
            
            reward = 100*((math.sin(3*obv[0]) * 0.0025 + 0.5 * obv[1] * obv[1]) - (math.sin(3*prevobv[0]) * 0.0025 + 0.5 * prevobv[1] * prevobv[1])) #Calculate our new reward using our reward function
            currentnode.AddObvTuple([prevobv, action, obv, (reward), 0]) #store the transition tuple in our state
            
            if done: #if the mountain car reaches the goal
                print("Splitting episode %d finished after %f time steps" % (episodes, timestep))
                break #break out of the current episode
            
        #update parameters for next episode
        explore_rate = get_explore_rate(episodes)
        learning_rate = get_learning_rate(episodes)

        if episodes%10 == 0: #if we have completed 10 episodes then we move onto the Expansion phase
                print("Splitting at episode %d" % (episodes))
                root.Processing()


def train():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  #the environment does not change over time so our discount factor is very high

    for episode in range(NUM_TRAIN_EPISODES): #For each episode        
        obv, _ = env.reset() #Reset our environment from the last episode
        currentnode = root.GetState(obv) #get our current state

        for t in range(MAX_TRAIN_TIME): #for each timestep
            env.render() #render the environment
            
            action = select_action(currentnode, explore_rate) #select an action
            prevobv = obv #log the previous observation
            obv, reward, done, _, _ = env.step(action) #execute the action
            currentnode = root.GetState(obv) #get our current state

            # Update the Q based on the result
            reward = 100*((math.sin(3*obv[0]) * 0.0025 + 0.5 * obv[1] * obv[1]) - (math.sin(3*prevobv[0]) * 0.0025 + 0.5 * prevobv[1] * prevobv[1])) #Calculate our new reward using our reward function
            currentnode.SetQ([prevobv, action, obv, (reward)], learning_rate) #update our Q

            if done: #If the car has reached the goal
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
        currentnode = root.GetState(obv) #get the current state

        for tt in range(0, MAX_TRAIN_TIME): #using a for loop for the purposes of efficiency
            env.render() #render the environment

            action = select_action(currentnode, 0) #select action
            obv, reward, done, _, _ = env.step(action) #execute action
            currentnode = root.GetState(obv) #get new state

            if done: #if the car reaches the goal over output the amount of time taken
                print("Test episode %d; time step %f." % (episode, tt))

        if tt < 200: #if the episode was successful
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
    plt.title("Continuous U-Tree Mountain Car Performance") #label the graph
    plt.show() #show the graph

if __name__ == "__main__": #This if statement means that if we import this file into another program it will not automatically run the code    
    start = timeit.default_timer() #program start time
    root = Node() #Initializing the root node
    print("Splitting")
    split()
    print('Training')
    train()
    print("Graphing")
    #graph()
    print('Testing')
    test()
    print("Number of splits total is: ", root.getsplit())
    print("Split index is: ", root.getsplitindex())
    stop = timeit.default_timer() #program finish time
    print('Total runtime for U-Tree Mountain Car is: ', stop - start, "seconds")
