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
from sklearn.cluster import DBSCAN
import numpy as np

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

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


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}
         

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    # inharit features from parents
    def __init__(self, *args, **kwargs):
        super(DefensiveReflexAgent, self).__init__(*args, **kwargs)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        print("Current position:", my_pos)

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
        # Implementation
        else:
            # Patrolling strategy when no invaders are visible
            features['patrol_distance'] = self.get_patrol_distance(successor) #changed it from game_state

        # Encoding the actions if we need to use it for rewards
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    

    # Defining patrol points
    def get_patrol_points(self, game_state):
        """
        Identify dynamic patrol points focusing on areas near remaining food
        and the nearest power capsule position.
        """
        patrol_points = []

        food_list = self.get_food(game_state).as_list()
        nearest_food_in_cluster = self.cluster_food(game_state, food_list)
        patrol_points.append(nearest_food_in_cluster)


        # Include additional strategic points like the nearest power capsule position
        power_capsule_position = self.get_power_capsule_position(game_state)
        if power_capsule_position:
            patrol_points.append(power_capsule_position) 

        return patrol_points
    
    #patrolling strategies
    def get_patrol_distance(self, game_state):
        """
        Calculate the average distance to key patrol points.
         """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        print("Current positionPatrol:", my_pos)

        # Define key patrol points (static or dynamically determined)
        patrol_points = self.get_patrol_points(game_state)

        # Calculate distances to each patrol point
        #distances = [self.get_maze_distance(tuple(my_pos), tuple(point)) for point in patrol_points] # point is a np.array, but it needs to be a tuple
        distances = [self.get_maze_distance(tuple(map(int, my_pos)), tuple(map(int, point))) for point in patrol_points]

        # Return the average distance
        if distances:
            return sum(distances) / len(distances)
        else:
            return 0
    

    def cluster_food(self, game_state, food_list, eps=3, min_samples=2):
        """
    Cluster food pellets using DBSCAN.

    :param food_list: List of food pellet coordinates.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: List of clusters with their food pellet coordinates.
        """
        # Convert food_list to a numpy array for DBSCAN
        food_array = np.array(food_list)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(food_array)

        # Extract clustered food pellets
        clusters = [food_array[dbscan.labels_ == label] for label in set(dbscan.labels_) if label != -1]

        if not clusters:
            return None

        # Find the largest cluster
        largest_cluster = max(clusters, key=len)

        # Get current position of the agent
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Find the nearest food in the largest cluster
        nearest_food = min(largest_cluster, key=lambda food: self.get_maze_distance(my_pos, tuple(food)))

        return tuple(nearest_food)


    def get_power_capsule_position(self, game_state):
        """
        Find and return the position of the nearest power capsule.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        capsules = game_state.get_capsules()

        if capsules:
            return min(capsules, key=lambda pos: self.get_maze_distance(my_pos, pos))
        else:
            return None
    

    def get_weights(self, game_state, action):
        """
    Dynamically adjust weights based on the current game state.
        """

        # Default weights
        weights = {
            'num_invaders': -1000, 
            'on_defense': 100, 
            'invader_distance': -10, 
            'stop': -100, 
            'reverse': -2,
            'patrol_distance': -5  # Weight for patrol distance
            }

        # Adjust weights based on specific game state conditions
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        # Example: Increase the penalty for stopping if there are invaders close by
        if invaders:
            closest_invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            if closest_invader_distance < 5:  # If an invader is very close
                weights['stop'] -= 50  # Increase the penalty for stopping

        # Example: Adjust weights based on the remaining food and moves
        remaining_food = len(self.get_food_you_are_defending(game_state).as_list())
        remaining_moves = game_state.data.timeleft
        total_moves = 1200  # Total moves before the game ends

        if remaining_food <= 4 or remaining_moves < total_moves / 4:
            weights['num_invaders'] *=3 
            weights['on_defense'] *= 2
            weights['patrol_distance'] *= 1

        return weights
    