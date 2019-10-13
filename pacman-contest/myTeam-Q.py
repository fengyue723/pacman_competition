# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util, json
from game import Directions
from game import Agent
from game import Actions
import game
import copy
import captureAgents


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.myPosition = gameState.getAgentPosition(self.index)
        self.teammatePosition = gameState.getAgentPosition((self.index+2)%4)
        print("index!!!:", self.index)
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.safeCells, self.dangerousCells = getSafeAndDangerousCells(gameState, self.height, self.width)
        self.homeBoundaryArea = getHomeBoundaryArea(self.width, self.height, gameState.getWalls(), self.red)
        self.enemyBoundaryArea = getEnemyBoundaryArea(self.width, self.height, gameState.getWalls(), self.red)
        self.semiDangerousCells_depth1, self.semiDangerousCells_depth2, self.dangerousCell_extreme =\
             getSemiDangerousCells(self.safeCells, self.dangerousCells)
        self.history = []
        # self.enemyWeight = 2
        # self.enemyWeightHistory = []
        self.shadowEnenmy = []
        self.leftStep = 300
        self.load = 20

        self.path = 'weights.json'
        self.alpha = 0.1
        self.discountRate = 0.9
        self.epsilon = 0.01
        self.lastQ = None
        self.lastReward = None
        self.lastFeature = None
        with open(self.path, "r") as f:
            self.weights = json.load(f)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getQValue(self, gameState, action):
        features, featureType, reward = self.getFeaturesAndReward(gameState, action)
        sumQ = 0
        for feature in features.keys():
            sumQ += features[feature] * self.weights[featureType][feature]
        return sumQ

    def getFeaturesAndReward(self, gameState, action):
        featureType = 'anyAction'
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        new_position = nextState.getAgentState(self.index).getPosition()
        # rewards
        reward = 0

        #features calculations
        nextRegionType = regionType(self.width, new_position, self.red)
        nextWeakEnemyLocation = self.weakEnemyLocation[:]
        nextStrongEnemyLocation = self.strongEnemyLocation[:]
        capsuleLocations = self.capsuleLocations[:]
        safeFood = self.safeFood[:]
        dangerousFood_depth1 = self.dangerousFood_depth1[:]
        dangerousFood_depth2 = self.dangerousFood_depth2[:]
        dangerousFood_extreme = self.dangerousFood_extreme[:]
        food_return = 0

        if new_position in safeFood:
            reward += 3
            nextFoodCarrying = self.foodCarrying+1
            safeFood.remove(new_position)
        elif new_position in dangerousFood_depth1:
            reward += 2.5
            nextFoodCarrying = self.foodCarrying+1
            dangerousFood_depth1.remove(new_position)
        elif new_position in dangerousFood_depth2:
            reward += 2
            nextFoodCarrying = self.foodCarrying+1
            dangerousFood_depth2.remove(new_position)
        elif new_position in dangerousFood_extreme:
            reward += 1.5
            nextFoodCarrying = self.foodCarrying+1
            dangerousFood_extreme.remove(new_position)
        else:
            nextFoodCarrying = self.foodCarrying

        if new_position in self.weakEnemyLocation:
            reward += 8
            nextWeakEnemyLocation.remove(new_position)


        homeDistance = 0
        if nextRegionType[0] or nextRegionType[1]:
            homeDistance = min([self.distancer.getDistance(new_position, b) for b in self.homeBoundaryArea])/10

        if homeDistance == 0:
            reward += self.foodCarrying*8
            food_return = self.foodCarrying

        if self.leftStep < homeDistance:
            reward -= 3

        if new_position in self.capsuleLocations:
            reward += 10
            capsuleLocations.remove(new_position)

        if self.strongEnemyLocation and \
            min([manhattanDistance(new_position, e) for e in self.strongEnemyLocation])<=1:
            reward -= 10

        
        # Nearest safe food:
        if safeFood:
            closest_Food = 1/min([self.distancer.getDistance(food, new_position) for food in safeFood])
        else:
            closest_Food = 0
        features['NearestSafeFood'] = closest_Food

        # Nearest dangerous food depth1:
        if dangerousFood_depth1:
            closest_Food = 1/min([self.distancer.getDistance(food, new_position) for food in dangerousFood_depth1])
        else:
            closest_Food = 0
        features['NearestDangerFood1'] = closest_Food

        # Nearest dangerous food depth2:
        if dangerousFood_depth2:
            closest_Food = 1/min([self.distancer.getDistance(food, new_position) for food in dangerousFood_depth2])
        else:
            closest_Food = 0
        features['NearestDangerFood2'] = closest_Food

        # Nearest extreme dangerous food:
        if dangerousFood_extreme:
            closest_Food = 1/min([self.distancer.getDistance(food, new_position) for food in dangerousFood_extreme])
        else:
            closest_Food = 0
        features['NearestDangerFood3'] = closest_Food


        # Nearest Capsule:
        if capsuleLocations:
            closest = 1/min([self.distancer.getDistance(c, new_position) for c in capsuleLocations])
        else:
            closest = 0
        features['NearestCapsule'] = closest            


        # Nearest Strong enemy
        if nextStrongEnemyLocation:
            closest = 1/min([self.distancer.getDistance(e, new_position) for e in nextStrongEnemyLocation])
        else:
            closest = 0
        features['NearestStrongEnemy'] = closest

        # Nearest Weak enemy
        if nextWeakEnemyLocation:
            closest = 1/min([self.distancer.getDistance(e, new_position) for e in nextWeakEnemyLocation])
        else:
            closest = 0
        features['NearestWeakEnemy'] = closest

        # Nearest Home distance:
        features['HomeDistance'] = homeDistance

        # food change:
        features['FoodReturn'] = food_return
        features['FoodCarrying'] = nextFoodCarrying

        #location type:
        features['LocType1'] = 1 if nextRegionType[0] else 0
        features['LocType2'] = 1 if nextRegionType[1] else 0
        features['LocType3'] = 1 if nextRegionType[2] else 0
        features['LocType4'] = 1 if nextRegionType[3] else 0

        if action == 'Stop':
            reward -= 500
            features['Stop'] = 1
        else:
            features['Stop'] = 0

        # repeated history check
        features['RepeatedHistory'] = repeatedHistory(self.history+[action])/10
        features['Bias'] = 1.0

        features.divideAll(100)

        return features, featureType, reward/100

    def updateWeights(self, Q, reward, Q_prime, features, featureType='anyAction'):
        additive = self.alpha * (reward + self.discountRate * Q_prime - Q)
        for feature in features.keys():
            self.weights[featureType][feature] += additive * features[feature]
        print("additive:", additive)
        with open(self.path, "w") as f:
            json.dump(self.weights, f)

    def chooseAction(self, gameState):
        with open(self.path, "r") as f:
            self.weights = json.load(f)
        self.myPosition = gameState.getAgentPosition(self.index)
        self.locType = regionType(self.width, self.myPosition, self.red)
        self.food = self.getFood(gameState)
        self.capsuleLocations = self.getCapsules(gameState)
        self.safeFood = getRelevantFood(self.food.asList(), self.safeCells)
        #self.dangerousFood = getRelevantFood(self.food.asList(), self.dangerousCells)
        self.dangerousFood_depth1 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth1)
        self.dangerousFood_depth2 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth2)
        self.dangerousFood_extreme = getRelevantFood(self.food.asList(), self.dangerousCell_extreme)
        self.capsuleMazedistance = [self.distancer.getDistance(c, self.myPosition) for c in self.capsuleLocations]

        self.enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        self.scaredTimer = gameState.getAgentState(self.index).scaredTimer
        if self.scaredTimer == 0:
            self.weakEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and e.scaredTimer>0]
            #不恐惧下，只害怕不恐惧的鬼
            self.strongEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and not e.isPacman and e.scaredTimer==0]
        else:
            self.weakEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and not e.isPacman and e.scaredTimer>0]
            #恐惧下，害怕自己家的pacman和不恐惧的鬼
            self.strongEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and (e.isPacman or\
                 ( not e.isPacman and e.scaredTimer == 0 ) )]



        self.foodCarrying = gameState.getAgentState(self.index).numCarrying




        actions = gameState.getLegalActions(self.index)

        # EpsGreedy
        values = []
        for action in actions:
            values.append((self.getQValue(gameState, action), action))
        best = random.choice([i for i in values if i == max(values)])
        if random.random() <= self.epsilon:
            #values.remove(best)
            best = random.choice(values)
        bestAction = best[1]

        #UPDATE!!
        if self.lastQ != None:
            self.updateWeights(self.lastQ, self.lastReward, max(values)[0], self.lastFeature)

        self.lastFeature, _, self.lastReward = self.getFeaturesAndReward(gameState, bestAction)
        self.lastQ = best[0]

        self.history.append(bestAction)
        self.history = self.history[-24:]

        print()
        print('features:', self.lastFeature)
        print()
        print('q:', self.lastQ)
        print()
        print('Reward:', self.lastReward)
        print()
        self.leftStep -= 1
        return bestAction



####################
# Helper functions #
####################
def getSafeAndDangerousCells(gameState, height, width):
  """
  A location is safe if it is on a circle.
  return:
  (circle, not_circle) #Both are set.
  """
  walls = gameState.getWalls().asList()
  path = set()
  for i in range(width):
    for j in range(height):
      if (i, j) not in walls:
        path.add((i, j))
  
  #DFS one time for finding all circles.
  d = {t:[] for t in path}
  cur_lis = [path.pop()]
  path.add(cur_lis[0])
  circle = set()

  while cur_lis:
    last = cur_lis[-1]
    has_child = False
    for child in [(last[0]+1,last[1]), (last[0]-1,last[1]), (last[0],last[1]+1), (last[0],last[1]-1)]:
      if child in path and child not in d[last] and (len(cur_lis)<2 or child != cur_lis[-2]):
        has_child = True
        d[last].append(child)
        if child in cur_lis:
          circle = circle.union(cur_lis[cur_lis.index(child):])
        else:
          cur_lis.append(child)
        break
    if has_child == False:
      cur_lis.pop()

  not_circle = path - circle
  return (circle, not_circle)

def repeatedHistory(history):
    if len(history)>3 and history[-4:-2]==history[-2:] and history[-1] != history[-2]:
        single = history[-2:]
        i = 2
        left = history[:-4]
        while len(left)>=2 and left[-2:] == single:
            i += 1
            left = left[:-2]
        return i
    elif len(history)>7 and history[-8:-4]==history[-4:] and history[-1] != history[-3]:
        single = history[-4:]
        i = 2
        left = history[:-8]
        while len(left)>=4 and left[-4:] == single:
            i += 1
            left = left[:-4]
        return i
    elif len(history)>11 and history[-12:-6]==history[-6:] and history[-1] != history[-4]:
        single = history[-6:]
        i = 2
        left = history[:-12]
        while len(left)>=6 and left[-6:] == single:
            i += 1
            left = left[:-6]
        return i
    else:
        return 1

def getSemiDangerousCells(safeCells, dangerousCells):
  dangerousFood_depth1 = []
  dangerousFood_depth2 = []
  dangerousFood_extreme = []
  for point in dangerousCells:
    if (point[0]+1, point[1]) in safeCells or (point[0], point[1]+1) in safeCells or \
      (point[0]-1, point[1]) in safeCells or (point[0], point[1]-1) in safeCells:
      dangerousFood_depth1.append(point)
  for point in dangerousCells:
    if point not in dangerousFood_depth1:
      if (point[0]+1, point[1]) in dangerousFood_depth1 or (point[0], point[1]+1) in dangerousFood_depth1 or \
        (point[0]-1, point[1]) in dangerousFood_depth1 or (point[0], point[1]-1) in dangerousFood_depth1:
        dangerousFood_depth2.append(point)
      else:
        dangerousFood_extreme.append(point)

  return dangerousFood_depth1, dangerousFood_depth2, dangerousFood_extreme


def getRelevantFood(foodList, Cells):
  """
  Return: Safe Food as List[tuple]
  """
  relevantFood = []
  for food in foodList:
    if food in Cells:
      relevantFood.append(food)
  return relevantFood




def regionType(width, position, isRed):
    """
    Input: width(int), position(x,y), isRed(boolean)
    Return: Bool[enemy/enemyBoundary, enemyBoundary, our/ourBoundary, ourBoundary]
    """
    enemy, enemyBoundary, our, ourBoundary = False, False, False, False
    boundary = int(width / 2)
    if (position[0] < boundary and isRed) or (position[0] >= boundary and not isRed):
        if position[0] == boundary-1 or position[0] == boundary:
            ourBoundary = True
        else:
            our = True
    else:
        if position[0] == boundary-1 or position[0] == boundary:
            enemyBoundary = True
        else:
            enemy = True
    return [enemy, enemyBoundary, our, ourBoundary]

def getHomeBoundaryArea(width, height, walls, isRed):
  if isRed:
    x = int(width/2)-1
  else:
    x = int(width/2)
  homeBoundaryArea = []
  for i in range(height):
    if not walls[x][i]:
      homeBoundaryArea.append((x, i))
  return homeBoundaryArea

def getEnemyBoundaryArea(width, height, walls, isRed):
  if isRed:
    x = int(width/2)
  else:
    x = int(width/2)-1
  enemyBoundaryArea = []
  for i in range(height):
    if not walls[x][i]:
      enemyBoundaryArea.append((x, i))
  return enemyBoundaryArea


def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])




####################
# Search functions #
####################


def aStarSearch(problem, heuristic, agent):
    """Search the node that has the lowest combined cost and heuristic first."""    

    heap = util.PriorityQueue()
    h0 = heuristic(problem.getStartState(), problem, agent)
    heap.push([problem.getStartState(), 0, None, None], [h0, h0])
    
    closed = {}
    #best_g = {}
    
    while not heap.isEmpty():
      node = heap.pop()
      g = node[1]
      node[0] = (node[0][0], tuple(node[0][1]))
      if node[0] not in closed: #or g < best_g[node[0]]:
        closed[node[0]] = [g, node[2], node[3]] #state:[g,action,father]
        #best_g[node[0]] = g

        if problem.isGoalState(node[0]):
          res = []
          cur = node[0]
          while closed[cur][2] != None:
            res.append(closed[cur][1])
            cur = closed[cur][2]
          res = res[::-1]
          return res

        for child in problem.getSuccessors((node[0][0], set(node[0][1]))):
          h = heuristic(child[0], problem, agent)
          g2 = g + child[2]
          heap.push([child[0], g2, child[1], node[0]], [g2+h,h])

    return []

def bfs_with_depth(problem, depth):    
    queue = util.Queue()
    queue.push([problem.getStartState(), 0])
    used = {problem.getStartState(): 0}
    while not queue.isEmpty():
      node = queue.pop()
      if node[1] < depth:
        for child in problem.getSuccessors(node[0]):
          if child not in used:
            used[child] = node[1]+1
            queue.push([child, node[1]+1])
    return list(used.keys())


