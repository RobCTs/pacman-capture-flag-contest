# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveGoodAgent', second='DefensiveGoodAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class GoodCaptureAgent(CaptureAgent):
    """
    A base class for agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        #self.numCarrying = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveGoodAgent(GoodCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    global numCarrying 
    numCarrying = 0 
    
    def getNumCarrying(self, game_state, action):
        currState = self.get_previous_observation() #current observation food list same as successor, gotta use previous  
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        if currState != None:
            curr_food_list = self.get_food(currState).as_list()
            #print("currFoodLength: ", len(curr_food_list))     
            #print("succFoodLength: ", len(food_list))
            diff = len(curr_food_list) - len(food_list)
            if 'numCarrying' in locals():
                numCarrying += diff 
                print("NUMCARRYING: ", numCarrying)
            if diff < 0: #negative difference means more food in future state that current state, i.e. pacman has died
                #reset nr of carrying food to 0
                numCarrying = 0 #num_succ
        else:
            #no previous observation available e.g. first/starting state
            numCarrying = 0
        if 'numCarrying' in locals() or 'numCarrying' in globals():
            return numCarrying
        return 0
    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        actions.remove(Directions.STOP) #again useless to get stuck
        values = []
        for a in actions:
            numCarrying = self.getNumCarrying(game_state, a)
            print("CHOOSE ACTION NUMCARRYING: ", numCarrying)
            v = self.MCTS(game_state, a, 5, 0, numCarrying)
            values.append(v)
        #print("Values1: ", values)
        #values = [self.evaluate(game_state, a, numCarrying) for a in actions]
        #print("Values2: ", values)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return bestActions[0]
    
    #to update global value from within a function, one has to pass it along every time
    def evaluate(self, game_state, action, numCarrying):
        print("evaluate NUMCARRYING: ", numCarrying)
        features = self.get_features(game_state, action, numCarrying)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    #TODO: possibly improve using ucb (https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/)
    def MCTS(self, game_state, action, depth, num_iters, numCarrying):
        print("MCTS NUMCARRYING: ", numCarrying)
        n = self.get_successor(game_state, action)
        actions = n.get_legal_actions(self.index)
        #print("actions: ", actions)
        actions.remove(Directions.STOP) #not doing anything is never a good choice
        if depth > 0:
            # random simulation 
            a = random.choice(actions)
            value = self.evaluate(game_state, action, numCarrying)
            # backpropagate and update parent nodes based on value of newly added curr child
            value = value + 0.2 ** num_iters * self.MCTS(n, a, depth - 1, num_iters + 1, numCarrying)
            return value
        else:
            return self.evaluate(game_state, action, numCarrying)
    
    def get_features(self, game_state, action, numCarrying):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        
        my_pos = successor.get_agent_state(self.index).get_position()
        
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            #print("MINDISTANCE: ", min_distance)
            features['distance_to_food'] = min_distance

        cap_list = self.get_capsules(successor)
        # Compute distance to the nearest power pellet/capsule
        if len(cap_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, c) for c in cap_list])
            features['distance_to_capsule'] = min_distance
        
        # Comput distance to the nearest opponent
        opp_index = self.get_opponents(successor)
        opponents_pos = []
        if len(opp_index) > 0:
            for i in opp_index:
                if not game_state.data.agent_states[i].is_pacman:
                    if game_state.data.agent_states[i].scared_timer < 5:
                        tmp = successor.get_agent_position(i)
                        if tmp != None: #get_agent_pos might return None
                            opponents_pos.append(tmp)
        #only if eligible opponents nearby (ghost and not scared for more than 5 more moves)
        if len(opponents_pos) > 0:    
            min_dist = min([self.get_maze_distance(my_pos, o) for o in opponents_pos])
            features['distance_to_opponent'] = min_dist
            
            #special death case (very time consuming so should be avoided most aka have highest weight)
            #if min_dist <=1:
            #    features['death'] = 1 #death = 1 if true
        else:
            features['distance_to_opponent'] = 0

         
        # Compute distance to nearest home tile/cell
        valid_homes = []
        # Consider where the home border is (e.g. to return after collecting food)
        # 2 walls on each side so width of actual home territory is layout-2
        w = game_state.data.layout.width
        mid_width = (w-2)/2
        if self.red == False: #blue territory starts to the right
            mid_width +=1
        h_height = game_state.data.layout.height
        
        for y in range(1, h_height):
            if not successor.has_wall(int(mid_width),y):
                valid_homes.append((int(mid_width),y))
        
        min_home = min([self.get_maze_distance(my_pos, h) for h in valid_homes])
        features['home'] = min_home
        #if 'min_dist' in locals(): #check if min_dist exists
        #    if min_dist < 1: #go home if enemy gets too close
        #        features['home'] = min_home*2

        # Consider the amount of food pacman is carrying
        currState = self.get_previous_observation() #current observation food list same as successor, gotta use previous  
        if currState != None:
            curr_food_list = self.get_food(currState).as_list()
            print("currFoodLength: ", len(curr_food_list))     
            print("succFoodLength: ", len(food_list))
            diff = len(curr_food_list) - len(food_list)
            
            numCarrying += diff 
            
            if diff < 0: #negative difference means more food in future state that current state, i.e. pacman has died
                #reset nr of carrying food to 0
                numCarrying = 0 #num_succ
        else:
            #no previous observation available e.g. first/starting state
            numCarrying = 0
        if numCarrying <=2:
            features['num_carrying'] = numCarrying
        else: 
            features['num_carrying'] = numCarrying/3 #force agent to go back more quickly when holding more food
        print("NUMCARRYING: ", numCarrying)
        return features

    def get_weights(self, game_state, action):
        
        #really force agent to return home consistently after carrying more than 3 food
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        
        if numCarrying >=3:
            return  {'successor_score': 5, 'distance_to_food': 0, 'distance_to_capsule': 0, 'distance_to_opponent': 1, 'home': -10,
                'num_carrying': 0}
        
        return {'successor_score': 10, 'distance_to_food': -2, 'distance_to_capsule': -1, 'distance_to_opponent': 1, 'home': -1,
                'num_carrying': 5}


class DefensiveGoodAgent(GoodCaptureAgent):
    """
    An agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
