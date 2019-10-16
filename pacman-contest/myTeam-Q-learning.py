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
               first='AttackerAgent', second='DefenderAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########
class DummyAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.myPosition = gameState.getAgentPosition(self.index)
        self.teammatePosition = gameState.getAgentPosition((self.index + 2) % 4)
        print("index!!!:", self.index)
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.safeCells, self.dangerousCells = getSafeAndDangerousCells(gameState, self.height, self.width)
        self.homeBoundaryArea = getHomeBoundaryArea(self.width, self.height, gameState.getWalls(), self.red)
        self.enemyBoundaryArea = getEnemyBoundaryArea(self.width, self.height, gameState.getWalls(), self.red)
        self.semiDangerousCells_depth1, self.semiDangerousCells_depth2, self.dangerousCell_extreme = \
            getSemiDangerousCells(self.safeCells, self.dangerousCells)
        self.history = []
        # self.enemyWeight = 2
        # self.enemyWeightHistory = []
        self.shadowEnenmy = []
        self.leftStep = 300
        self.load = 20

        self.alpha = 0.1
        self.discountRate = 0.9
        self.epsilon = 0.00001
        self.lastQ = None
        self.lastReward = None
        self.lastFeature = None
        self.path = 'weights.json'
        self.weights = {}

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
        return {}, 'anyAction', 0

    def updateWeights(self, Q, reward, Q_prime, features, featureType='anyAction'):
        additive = self.alpha * (reward + self.discountRate * Q_prime - Q)
        for feature in features.keys():
            self.weights[featureType][feature] += additive * features[feature]
        print("additive:", additive)
        with open(self.path, "w") as f:
            json.dump(self.weights, f)
            # time.sleep(1)


class AttackerAgent(DummyAgent):

    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)
        # self.path = 'weights-a.json'
        # with open(self.path, "r") as f:
        #     self.weights = json.load(f)
        self.weights = {
            "anyAction": {
                "NearestSafeFood": -0.07039863411379269,
                "NearestDangerFood1": -0.0020792651949881067,
                "NearestDangerFood2": -0.004706231804810953,
                "NearestDangerFood3": 0.009681747375314654,
                "NearestCapsule": -0.0036485591952501533,
                "NearestStrongEnemy": 0.05110989185848645,
                "NearestWeakEnemy": -0.010806867784807321,
                "HomeDistance": 0.08527301783256504,
                "HomeDistance2": 0.00039406202685385544,
                "Stop": -0.45472336668549174,
                "FoodReturn": 0.11617826390053419,
                "FoodCarrying": 0.33717588260738557,
                "LocType1": 0.04722334716423313,
                "LocType2": 0.00013286636882859764,
                "LocType3": -0.021735313730104792,
                "LocType4": -0.006516915039130188,
                "RepeatedHistory": -0.018783828090497386,
                "danger": -1.24201175332375,
                "load&distance": -0.06897212443025055,
                "Bias": 0.1910398476382674
            }
        }

    def getFeaturesAndReward(self, gameState, action):
        featureType = 'anyAction'
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        new_position = nextState.getAgentState(self.index).getPosition()
        # rewards
        reward = 0

        # features calculations
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
            reward += 6
            nextFoodCarrying = self.foodCarrying + 1
            # safeFood.remove(new_position)
        elif new_position in dangerousFood_depth1:
            reward += 5
            nextFoodCarrying = self.foodCarrying + 1
            # dangerousFood_depth1.remove(new_position)
        elif new_position in dangerousFood_depth2:
            reward += 4
            nextFoodCarrying = self.foodCarrying + 1
            # dangerousFood_depth2.remove(new_position)
        elif new_position in dangerousFood_extreme:
            reward += 3
            nextFoodCarrying = self.foodCarrying + 1
            # dangerousFood_extreme.remove(new_position)
        else:
            nextFoodCarrying = self.foodCarrying

        if new_position in self.weakEnemyLocation:
            reward += 8
            # nextWeakEnemyLocation.remove(new_position)

        homeDistance = 0
        if nextRegionType[0] or nextRegionType[1]:
            homeDistance = min([self.distancer.getDistance(new_position, b) for b in self.homeBoundaryArea]) / 10

        if homeDistance == 0:
            reward += self.foodCarrying * 4
            food_return = self.foodCarrying

        if self.leftStep < homeDistance * 10:
            reward -= 3
            homeDistance2 = homeDistance - 1
        else:
            homeDistance2 = 0

        if new_position in self.capsuleLocations:
            reward += 15
            # capsuleLocations.remove(new_position)

        if self.strongEnemyLocation and \
                min([manhattanDistance(new_position, e) for e in self.strongEnemyLocation]) <= 1:
            danger = 1
            reward -= 25
        else:
            danger = 0

        # Nearest safe food:
        if safeFood:
            closest_Food = min([self.distancer.getDistance(food, new_position) for food in safeFood]) / 100
        else:
            closest_Food = 0.2
        features['NearestSafeFood'] = closest_Food

        # Nearest dangerous food depth1:
        if dangerousFood_depth1:
            closest_Food = min([self.distancer.getDistance(food, new_position) for food in dangerousFood_depth1]) / 100
        else:
            closest_Food = 0.2
        features['NearestDangerFood1'] = closest_Food

        # Nearest dangerous food depth2:
        if dangerousFood_depth2:
            closest_Food = min([self.distancer.getDistance(food, new_position) for food in dangerousFood_depth2]) / 100
        else:
            closest_Food = 0.2
        features['NearestDangerFood2'] = closest_Food

        # Nearest extreme dangerous food:
        if dangerousFood_extreme:
            closest_Food = min([self.distancer.getDistance(food, new_position) for food in dangerousFood_extreme]) / 100
        else:
            closest_Food = 0.2
        features['NearestDangerFood3'] = closest_Food

        # Nearest Capsule:
        if capsuleLocations:
            closest = min([self.distancer.getDistance(c, new_position) for c in capsuleLocations]) / 100
        else:
            closest = 0.2
        features['NearestCapsule'] = closest

        # Nearest Strong enemy
        if nextStrongEnemyLocation:
            closest = min([self.distancer.getDistance(e, new_position) for e in nextStrongEnemyLocation]) / 100
        else:
            closest = 0.2
        features['NearestStrongEnemy'] = closest

        # Nearest Weak enemy
        if nextWeakEnemyLocation:
            closest = min([self.distancer.getDistance(e, new_position) for e in nextWeakEnemyLocation]) / 100
        else:
            closest = 0.2
        features['NearestWeakEnemy'] = closest

        # Nearest Home distance:
        features['HomeDistance'] = homeDistance
        features['HomeDistance2'] = homeDistance2

        # food change:
        features['FoodReturn'] = food_return
        features['FoodCarrying'] = nextFoodCarrying

        # location type:
        features['LocType1'] = 0.1 if nextRegionType[0] else 0
        features['LocType2'] = 0.1 if nextRegionType[1] else 0
        features['LocType3'] = 0.1 if nextRegionType[2] else 0
        features['LocType4'] = 0.1 if nextRegionType[3] else 0

        if action == 'Stop':
            reward -= 5
            features['Stop'] = 1
        else:
            features['Stop'] = 0

        # repeated history check
        features['RepeatedHistory'] = repeatedHistory(self.history + [action]) / 10
        reward -= (features['RepeatedHistory'] - 0.1) * 8
        features['Bias'] = 1.0
        features['danger'] = danger
        features['load&distance'] = homeDistance * nextFoodCarrying / 10

        features.divideAll(10)

        return features, featureType, reward / 100

    def chooseAction(self, gameState):
        # with open(self.path, "r") as f:
        #     self.weights = json.load(f)
        self.myPosition = gameState.getAgentPosition(self.index)
        self.locType = regionType(self.width, self.myPosition, self.red)
        self.food = self.getFood(gameState)
        self.capsuleLocations = self.getCapsules(gameState)
        self.safeFood = getRelevantFood(self.food.asList(), self.safeCells)
        # self.dangerousFood = getRelevantFood(self.food.asList(), self.dangerousCells)
        self.dangerousFood_depth1 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth1)
        self.dangerousFood_depth2 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth2)
        self.dangerousFood_extreme = getRelevantFood(self.food.asList(), self.dangerousCell_extreme)
        self.capsuleMazedistance = [self.distancer.getDistance(c, self.myPosition) for c in self.capsuleLocations]

        self.enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        self.scaredTimer = gameState.getAgentState(self.index).scaredTimer
        if self.scaredTimer == 0:
            self.weakEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and e.scaredTimer > 0]
            # Not at scare time, only avoiding the ghost
            self.strongEnemyLocation = [e.getPosition() for e in self.enemies if
                                        e.getPosition() and not e.isPacman and e.scaredTimer == 0]
        else:
            self.weakEnemyLocation = [e.getPosition() for e in self.enemies if
                                      e.getPosition() and not e.isPacman and e.scaredTimer > 0]
            # Not at scare time, only avoiding the ghost and the opponent's pacman
            self.strongEnemyLocation = [e.getPosition() for e in self.enemies if e.getPosition() and (e.isPacman or \
                                                                                                      (
                                                                                                              not e.isPacman and e.scaredTimer == 0))]

        self.foodCarrying = gameState.getAgentState(self.index).numCarrying

        actions = gameState.getLegalActions(self.index)

        # EpsGreedy
        values = []
        for action in actions:
            values.append((self.getQValue(gameState, action), action))
        best = random.choice([i for i in values if i == max(values)])
        if random.random() <= self.epsilon:
            # values.remove(best)
            best = random.choice(values)
        bestAction = best[1]

        # UPDATE!!
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


class DefenderAgent(DummyAgent):
    def registerInitialState(self, gameState):
        DummyAgent.registerInitialState(self, gameState)
        self.state = gameState.getAgentState(self.index)
        self.bestHomeBoundaryArea = findEntrance(self, gameState)
        self.target = list()
        # self.path = 'weights-d.json'
        # with open(self.path, "r") as f:
        #     self.weights = json.load(f)
        self.weights = {
            "anyAction": {
                "ClosestWeakInvader": 9.058505863910638,
                "ClosestStrongInvader": -2.0,
                "InvaderSearch": 29.17467385311651,
                "HomeDistance": -9.285935561728046e-05,
                "Stop": -7.849895680723872,
                "RepeatedHistory": -3.732388650550977,
                "Bias": -26.138459317208422
            }
        }

    def getFeaturesAndReward(self, gameState, action):
        featureType = 'anyAction'
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        new_position = nextState.getAgentState(self.index).getPosition()
        # rewards
        reward = 0

        # features calculations
        weakInvaderLocation = self.weakInvaderLocation
        strongInvaderLocation = self.strongInvaderLocation
        target = self.target

        if new_position in weakInvaderLocation:
            reward += 20
        if new_position in strongInvaderLocation:
            reward -= 20
        if new_position in target:
            reward += 20
        if not self.state.isPacman:
            reward -= 100
        if (new_position in self.bestHomeBoundaryArea) and len(self.invader) == 0:
            reward += 1
        elif (new_position in self.bestHomeBoundaryArea) and len(self.invader) > 0:
            reward -= 1

        # homeDistance = 0
        # if nextRegionType[0] or nextRegionType[1]:
        #     homeDistance = min([self.distancer.getDistance(new_position, b) for b in self.homeBoundaryArea])/10

        # if self.strongEnemyLocation and \
        #     min([manhattanDistance(new_position, e) for e in self.strongEnemyLocation])<=1:
        #     reward -= 10

        closest = 0
        # Nearest weak invader:
        if len(weakInvaderLocation) > 0:
            closest = 1 / (
                    min([self.distancer.getDistance(invader, new_position) for invader in weakInvaderLocation]) + 1)
        else:
            closest = 0
        features['ClosestWeakInvader'] = closest

        # Nearest strong invader:
        if len(strongInvaderLocation) > 0:
            closest = 1 / (min(
                [self.distancer.getDistance(invader, new_position) for invader in strongInvaderLocation]) + 1)
        else:
            closest = 0
        features['ClosestStrongInvader'] = closest

        # Go to last eaten food
        if len(target) > 0:
            closest = 1 / (min([self.distancer.getDistance(food, new_position) for food in target]) + 1)
        else:
            closest = 0
        features['InvaderSearch'] = closest

        # Go to last eaten food
        if len(self.bestHomeBoundaryArea) > 0:
            closest = 1 / (min(
                [self.distancer.getDistance(location, new_position) for location in self.bestHomeBoundaryArea]) + 1)
        else:
            closest = 0
        features['HomeDistance'] = closest

        # # Nearest Home distance:
        # features['HomeDistance'] = homeDistance

        if action == 'Stop':
            reward -= 100
            features['Stop'] = 1
        else:
            features['Stop'] = 0

        # repeated history check
        features['RepeatedHistory'] = repeatedHistory(self.history + [action]) / 10
        features['Bias'] = 1.0

        features.divideAll(100)

        return features, featureType, reward / 100

    def chooseAction(self, gameState):
        # with open(self.path, "r") as f:
        #     self.weights = json.load(f)
        self.myPosition = gameState.getAgentPosition(self.index)
        self.locType = regionType(self.width, self.myPosition, self.red)
        self.food = self.getFood(gameState)
        self.capsuleLocations = self.getCapsules(gameState)
        self.safeFood = getRelevantFood(self.food.asList(), self.safeCells)
        # self.dangerousFood = getRelevantFood(self.food.asList(), self.dangerousCells)
        self.capsuleMazedistance = [self.distancer.getDistance(c, self.myPosition) for c in self.capsuleLocations]

        self.scaredTimer = gameState.getAgentState(self.index).scaredTimer

        lastEatenFood = findLastEatenFood(self, gameState)
        if lastEatenFood:
            self.target = lastEatenFood

        self.enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        self.invader = [e.getPosition() for e in self.enemies if e.getPosition() and e.isPacman]
        self.weakInvaderLocation = []
        self.strongInvaderLocation = []
        if self.state.scaredTimer == 0:
            self.weakInvaderLocation = [e.getPosition() for e in self.enemies if e.getPosition() and e.isPacman]
        else:
            self.strongInvaderLocation = [[e.getPosition() for e in self.enemies if e.getPosition() and e.isPacman]]

        # self.foodCarrying = gameState.getAgentState(self.index).numCarrying

        actions = gameState.getLegalActions(self.index)

        # EpsGreedy
        values = []
        for action in actions:
            values.append((self.getQValue(gameState, action), action))
        best = random.choice([i for i in values if i == max(values)])
        if random.random() <= self.epsilon:
            # values.remove(best)
            best = random.choice(values)
        bestAction = best[1]

        # UPDATE!!
        if self.lastQ != None:
            self.updateWeights(self.lastQ, self.lastReward, max(values)[0], self.lastFeature)

        self.lastFeature, _, self.lastReward = self.getFeaturesAndReward(gameState, bestAction)
        self.lastQ = best[0]

        self.history.append(bestAction)
        self.history = self.history[-24:]

        nextState = self.getSuccessor(gameState, action)
        new_position = nextState.getAgentState(self.index).getPosition()
        if new_position in self.target:
            self.target.remove(new_position)

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

    # DFS one time for finding all circles.
    d = {t: [] for t in path}
    cur_lis = [path.pop()]
    path.add(cur_lis[0])
    circle = set()

    while cur_lis:
        last = cur_lis[-1]
        has_child = False
        for child in [(last[0] + 1, last[1]), (last[0] - 1, last[1]), (last[0], last[1] + 1), (last[0], last[1] - 1)]:
            if child in path and child not in d[last] and (len(cur_lis) < 2 or child != cur_lis[-2]):
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
    if len(history) > 3 and history[-4:-2] == history[-2:] and history[-1] != history[-2]:
        single = history[-2:]
        i = 2
        left = history[:-4]
        while len(left) >= 2 and left[-2:] == single:
            i += 1
            left = left[:-2]
        return i
    elif len(history) > 7 and history[-8:-4] == history[-4:] and history[-1] != history[-3]:
        single = history[-4:]
        i = 2
        left = history[:-8]
        while len(left) >= 4 and left[-4:] == single:
            i += 1
            left = left[:-4]
        return i
    elif len(history) > 11 and history[-12:-6] == history[-6:] and history[-1] != history[-4]:
        single = history[-6:]
        i = 2
        left = history[:-12]
        while len(left) >= 6 and left[-6:] == single:
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
        if (point[0] + 1, point[1]) in safeCells or (point[0], point[1] + 1) in safeCells or \
                (point[0] - 1, point[1]) in safeCells or (point[0], point[1] - 1) in safeCells:
            dangerousFood_depth1.append(point)
    for point in dangerousCells:
        if point not in dangerousFood_depth1:
            if (point[0] + 1, point[1]) in dangerousFood_depth1 or (point[0], point[1] + 1) in dangerousFood_depth1 or \
                    (point[0] - 1, point[1]) in dangerousFood_depth1 or (
                    point[0], point[1] - 1) in dangerousFood_depth1:
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
        if position[0] == boundary - 1 or position[0] == boundary:
            ourBoundary = True
        else:
            our = True
    else:
        if position[0] == boundary - 1 or position[0] == boundary:
            enemyBoundary = True
        else:
            enemy = True
    return [enemy, enemyBoundary, our, ourBoundary]


def getHomeBoundaryArea(width, height, walls, isRed):
    if isRed:
        x = int(width / 2) - 1
    else:
        x = int(width / 2)
    homeBoundaryArea = []
    for i in range(height):
        if not walls[x][i]:
            homeBoundaryArea.append((x, i))
    return homeBoundaryArea


def getEnemyBoundaryArea(width, height, walls, isRed):
    if isRed:
        x = int(width / 2)
    else:
        x = int(width / 2) - 1
    enemyBoundaryArea = []
    for i in range(height):
        if not walls[x][i]:
            enemyBoundaryArea.append((x, i))
    return enemyBoundaryArea


def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def findLastEatenFood(agent, gameState):
    res = list()
    if len(agent.observationHistory) > 1:
        previousState = agent.getPreviousObservation()
        previousFood = agent.getFoodYouAreDefending(previousState).asList()
        currentFood = agent.getFoodYouAreDefending(gameState).asList()
        for food in previousFood:
            if food not in currentFood:
                res.append(food)
    return res


def findEntrance(agent, gameState):
    res = sorted(agent.homeBoundaryArea, key=lambda x: x[1])
    results = list()
    length = len(res)

    lastBreak = 0
    if length > 1:
        lastY = res[0][1]
        curLen = 1
        for i in range(1, length):
            y = res[i][1]
            if y == lastY + 1:
                curLen += 1
            else:
                results.append(res[lastBreak + curLen // 2])
                lastBreak = i
                curLen = 1
            lastY = y
        results.append(res[lastBreak + curLen // 2])
        return results
    else:
        return res


####################
# Search functions #
####################


def aStarSearch(problem, heuristic, agent):
    """Search the node that has the lowest combined cost and heuristic first."""

    heap = util.PriorityQueue()
    h0 = heuristic(problem.getStartState(), problem, agent)
    heap.push([problem.getStartState(), 0, None, None], [h0, h0])

    closed = {}
    # best_g = {}

    while not heap.isEmpty():
        node = heap.pop()
        g = node[1]
        node[0] = (node[0][0], tuple(node[0][1]))
        if node[0] not in closed:  # or g < best_g[node[0]]:
            closed[node[0]] = [g, node[2], node[3]]  # state:[g,action,father]
            # best_g[node[0]] = g

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
                heap.push([child[0], g2, child[1], node[0]], [g2 + h, h])

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
                    used[child] = node[1] + 1
                    queue.push([child, node[1] + 1])
    return list(used.keys())
