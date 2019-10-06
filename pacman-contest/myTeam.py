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
import random, time, util
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
               first = 'DefenderAgent', second = 'ReaperAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
    self.start = gameState.getAgentPosition(self.index)
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    
    

  def chooseAction(self, gameState):
    """
    Picks among actions by scores.
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.distancer.getDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    '''
    You should change this in your own agent.
    '''

    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 100, 'distanceToFood': -1}


class ReaperAgent(DummyAgent):
  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)
    self.initialPosition = gameState.getAgentPosition(self.index)
    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width
    self.safeCells, self.dangerousCells = getSafeAndDangerousCells(gameState, self.height, self.width)
    self.semiDangerousCells_depth1, self.semiDangerousCells_depth2 = getSemiDangerousCells(self.safeCells, self.dangerousCells)
    self.homeBoundaryArea = getHomeBoundaryArea(self.width, self.height, gameState.getWalls(), self.red)
    self.history = []
    self.enemyWeight = 2
    self.enemyWeightHistory = []
    self.shadowEnenmy = []

  def chooseAction(self, gameState):
    print()
    print('Reaper new round:')
    if self.getPreviousObservation() and \
      self.distancer.getDistance(self.getPreviousObservation().getAgentPosition(self.index),\
                                          gameState.getAgentPosition(self.index))!=1:
      self.enemyWeight = 2
    elif self.enemyWeight == 3 and repeatedHistory(self.history):
      self.enemyWeight = 4
    elif self.enemyWeight == 2 and repeatedHistory(self.history):
      self.enemyWeight = 3
    elif self.enemyWeight == 1 and repeatedHistory(self.history):
      self.enemyWeight = 2
    elif self.enemyWeight >= 4 and repeatedHistory(self.history):
      self.enemyWeight = min(self.enemyWeight+1, 7)
    elif len(self.enemyWeightHistory) >5 and len(set(self.enemyWeightHistory[-6:])) == 1:
      self.enemyWeight = max(self.enemyWeight-1, 2)

    self.myPosition = gameState.getAgentPosition(self.index)
    self.food = self.getFood(gameState)

    self.safeFood = getRelevantFood(self.food.asList(), self.safeCells)
    #self.dangerousFood = getRelevantFood(self.food.asList(), self.dangerousCells)
    self.dangerousFood_depth1 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth1)
    self.dangerousFood_depth2 = getRelevantFood(self.food.asList(), self.semiDangerousCells_depth2)

    #print("safefood:", self.safeFood)

    enemyLocation = []
    enemyshadowLocation = []
    enemyDistance = [] #Observable oppenents distances
    for enemy in self.getOpponents(gameState):
      enemyPosition = gameState.getAgentPosition(enemy)
      if enemyPosition:
        enemyLocation.append(enemyPosition)
        enemyDistance.append(self.distancer.getDistance(self.myPosition, enemyPosition))

    if len(enemyLocation)<1:
      enemyLocation = self.shadowEnenmy
      for enemyPosition in enemyLocation:
        enemyDistance.append(self.distancer.getDistance(self.myPosition, enemyPosition))
    elif len(enemyLocation) == 2:
      self.shadowEnenmy = [e for e in enemyLocation if manhattanDistance(e, self.myPosition) == 5 \
        or manhattanDistance(e, self.myPosition) == 4]
    elif len(enemyLocation) == 1:
      cur_distance = manhattanDistance(enemyLocation[0], self.myPosition)
      if cur_distance == 5 or cur_distance == 4:
        self.shadowEnenmy = [shadow for shadow in self.shadowEnenmy \
          if manhattanDistance(shadow, enemyLocation[0]) > 2]
        self.shadowEnenmy.extend(enemyLocation)
        if len(self.shadowEnenmy)>2:
          self.shadowEnenmy = sorted(self.shadowEnenmy, key=lambda x:manhattanDistance(x, self.myPosition))[:2]
        enemyLocation = self.shadowEnenmy
      else:
        self.shadowEnenmy = [shadow for shadow in self.shadowEnenmy \
          if manhattanDistance(shadow, enemyLocation[0]) > 2]
        enemyLocation.extend(self.shadowEnenmy)

    print("enemyLocation", enemyLocation)
    print("shadowEnenmy", self.shadowEnenmy)



    


    if enemyDistance:
      minOppenentDistance = min(enemyDistance)
      # if minOppenentDistance == 2:
      #   self.enemyWeight = max(self.enemyWeight, 2)
      # elif minOppenentDistance == 1 and self.history[-1] == 'Stop':
      #   self.enemyWeight = 1
      # nearestEnemy = [a for a, v in zip(enemyLocation, enemyDistance) if v == min(enemyDistance)][0]
      # if self.enemyWeight > 4:
      #   for i in range(self.enemyWeight-2):
      #     # enemyshadowLocation.append((nearestEnemy[0]+i, nearestEnemy[1]))
      #     # enemyshadowLocation.append((nearestEnemy[0]-i, nearestEnemy[1]))
      #     if self.distancer.getDistance((nearestEnemy[0], nearestEnemy[1]+i), self.myPosition) < i:
      #       enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]+i))
      #     if self.distancer.getDistance((nearestEnemy[0], nearestEnemy[1]+i), self.myPosition) < i:
      #       enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]+i))
      # elif self.enemyWeight == 4:
      #   enemyshadowLocation.append((nearestEnemy[0]+1, nearestEnemy[1]))
      #   enemyshadowLocation.append((nearestEnemy[0]-1, nearestEnemy[1]))
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]+1))
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]-1))
      #   enemyshadowLocation.append((nearestEnemy[0]+1, nearestEnemy[1]+1))
      #   enemyshadowLocation.append((nearestEnemy[0]-1, nearestEnemy[1]+1))
      #   enemyshadowLocation.append((nearestEnemy[0]+1, nearestEnemy[1]-1))
      #   enemyshadowLocation.append((nearestEnemy[0]-1, nearestEnemy[1]-1))
      # elif self.enemyWeight == 3:
      #   enemyshadowLocation.append((nearestEnemy[0]+1, nearestEnemy[1]))
      #   enemyshadowLocation.append((nearestEnemy[0]-1, nearestEnemy[1]))
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]+1))
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]-1))
      # elif self.enemyWeight == 2:
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]+1))
      #   enemyshadowLocation.append((nearestEnemy[0], nearestEnemy[1]-1))
    else:
      minOppenentDistance = None
    
    self.enemyWeightHistory.append(self.enemyWeight)
    self.enemyWeightHistory = self.enemyWeightHistory[-6:]

    #Decisions
    
    heuristic = foodHeuristic2 if len(self.food.asList()) < 50 and self.food.asList() and \
      min([self.distancer.getDistance(v, self.myPosition) for v in self.food.asList()])<11 else foodHeuristic1

    print("enemyWeight:", self.enemyWeight)
    self.locType = regionType(self.width, self.myPosition, self.red)
    carrying = gameState.getAgentState(self.index).numCarrying

    while True:
      """
      Eat all if foodleft >2 and
         1. In home safe region
      or 2. No enemy insight or far away from me and numCarrying < 3
      or 3. No safe food anymore and no food carrying
      or 4. Enemy scared time > 8
      """
      if len(self.food.asList())>2 and ( (carrying < 3 and (self.locType[2] or \
        not minOppenentDistance or minOppenentDistance>=7)) or ( len(self.dangerousFood_depth2) == 0 and\
          len(self.dangerousFood_depth1) == 0 and len(self.safeFood) == 0 and \
            carrying == 0) or gameState.getAgentState(self.getOpponents(gameState)[0]).scaredTimer>8 ): 
        print("eat all!")
        if gameState.getAgentState(self.index).scaredTimer == 0:
          problem = SearchNearestProblem(gameState, self, [e for e in enemyLocation if \
            regionType(self.width, e, self.red)[0] or regionType(self.width, e, self.red)[1]], self.enemyWeight)
        else:
          problem = SearchNearestProblem(gameState, self, enemyLocation, self.enemyWeight)
        actions = aStarSearch(problem, heuristic, self)
        if actions:
          break
      
      """
      Eat Danger_depth2 if foodleft > 2 and carry <= 10 and one of:
      1. no enemy insight
      2. enmeyDistance >= 5
      3. no more Safer foods and no food carrying
      """
      if len(self.food.asList())>2 and carrying <= 10 and ( not minOppenentDistance or minOppenentDistance>=5 or ( \
          len(self.dangerousFood_depth1) == 0 and len(self.safeFood) == 0 and carrying == 0) ):
        print("eat danger2!")
        problem = SearchDangerProblem_depth2(gameState, self, enemyLocation, self.enemyWeight)
        actions = aStarSearch(problem, foodHeuristic1, self)
        if actions:
          break

      """
      Eat Danger_depth1 if foodleft > 2 and carry <= 10 and one of:
      1. no enemy insight
      2. enmeyDistance >= 3
      3. no more Safer foods and no food carrying
      """
      if len(self.food.asList())>2 and carrying <= 10 and ( not minOppenentDistance or \
        minOppenentDistance>=3 or (len(self.safeFood) == 0 and carrying == 0) ):
        print("eat danger1!")
        problem = SearchDangerProblem_depth1(gameState, self, enemyLocation, self.enemyWeight)
        actions = aStarSearch(problem, foodHeuristic1, self)
        if actions:
          break

      """
      Eat Danger_depth1 if foodleft > 2 and one of:
      1. food carrying < max(left safefood, 5)
      """
      if len(self.food.asList())>2 and carrying < max(min(len(self.safeFood), 15), 5):
        print('eat safe!')
        problem = SearchSafeFoodProblem(gameState, self, enemyLocation, self.enemyWeight)
        actions = aStarSearch(problem, heuristic, self)
        if actions:
          break

      print("going home!")
      problem = GoHomeProblem(gameState, self, enemyLocation, self.enemyWeight)
      actions = aStarSearch(problem, foodHeuristic1, self)
      if not actions:
        problem = GoHomeProblem(gameState, self, [], self.enemyWeight)
        actions = aStarSearch(problem, foodHeuristic1, self)

      break

    if actions:
      self.history.append(actions[0])
      print(actions)
      return actions[0]
    else:
      if self.history[-1] == 'Stop':
        action =  random.choice(gameState.getLegalActions(self.index))
      else:
        action = 'Stop'
      self.history.append(action)
      self.history = self.history[-6:]
      print(action)
      return action

class DefenderAgent(DummyAgent):
  """
  Agent mainly in charge of denfending our food
  """

  def registerInitialState(self, gameState):
    DummyAgent.registerInitialState(self, gameState)
    self.minX = 0 if self.red else gameState.data.layout.width/2
    self.maxX = gameState.data.layout.width/2-1 if self.red else gameState.data.layout.width - 1
    self.walls = set(gameState.getWalls().asList())
    self.target = list()
    self.lastChaseTarget = list()
  
  def chooseAction(self, gameState):
    print()
    print('defender new round:')
    # print(gameState.getLegalActions(self.index))
    self.myPosition = gameState.getAgentPosition(self.index)
    # Computes distance to invaders we can see
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemiesInSight = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    invaders = [invader for invader in enemies if invader.isPacman]
    invadersInSight = [invader for invader in invaders if invader.getPosition() != None]
    self.lastEatenFood = self.findLastEatenFood(gameState)
    self.target = list()

    if self.myPosition in self.lastChaseTarget:
      self.lastChaseTarget.clear()

    if len(invaders) > 0:
      print('invaders:' + str(len(invaders)))
      # Chase the closest invader in sight 入侵者在视野范围内
      if len(invadersInSight) > 0 and gameState.getAgentState(self.index).scaredTimer == 0:
        distance = 99999999
        position = tuple()
        for invader in invadersInSight:
          if distance > self.distancer.getDistance(self.myPosition, invader.getPosition()):
            distance = self.distancer.getDistance(self.myPosition, invader.getPosition())
            position = invader.getPosition()
        self.target.append(position)
        self.lastChaseTarget = self.target
        problem = GoToProblem(gameState, self)
        actions = aStarSearch(problem, foodHeuristic1, self)
        return actions[0] if len(actions) > 0 else 'Stop'

      # Find invaders in our land 不知道入侵者在哪，但有食物被吃了 -> 去被吃食物附近
      if len(invadersInSight) == 0 and len(self.lastEatenFood) != 0 and gameState.getAgentState(self.index).scaredTimer == 0:
        self.target = self.lastEatenFood
        self.lastChaseTarget = self.target
        problem = GoToProblem(gameState, self)
        actions = aStarSearch(problem, foodHeuristic1, self)
        return actions[0] if len(actions) > 0 else 'Stop'

      if len(self.target) == 0 and len(self.lastChaseTarget) > 0:
        self.target = self.lastChaseTarget
        problem = GoToProblem(gameState, self)
        actions = aStarSearch(problem, foodHeuristic1, self)
        return actions[0] if len(actions) > 0 else 'Stop'

      self.target = self.findEntrance(gameState) # 没入侵者 -> 去入口等
      if gameState.getAgentPosition(self.index) in self.target: 
        return 'Stop'
      problem = GoToProblem(gameState, self)
      actions = aStarSearch(problem, foodHeuristic1, self)
      return actions[0] if len(actions) > 0 else 'Stop'

    if len(invaders) == 0:
      print('invaders:' + str(len(invaders)))
      if len(enemiesInSight) > 0:
        print('enemiesInSight:' + str(len(enemiesInSight)))
        tempActions = self.getLegalActionsNoCrossing(gameState)
        distance = 9999999
        res = list()
        for action in tempActions:
          successor = self.getSuccessor(gameState, action)
          for enemy in enemiesInSight:
            cur = self.distancer.getDistance(successor.getAgentPosition(self.index), enemy.getPosition())
            if distance > cur:
              distance = cur
              res.clear()
              res.append(action)
            if distance == cur:
              res.append(action)
        print(res)
        return random.choice(res)

      else:
        print('enemiesInSight:' + str(len(enemiesInSight)))
        self.target = self.findEntrance(gameState) # 没入侵者 -> 去入口等
        if gameState.getAgentPosition(self.index) in self.target: 
          return 'Stop'
        problem = GoToProblem(gameState, self)
        actions = aStarSearch(problem, foodHeuristic1, self)
        return actions[0] if len(actions) > 0 else 'Stop'

    # baseline defender
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self,gameState, action):
    return {'numInvaders': -1, 
            'onDefense': 200, 
            'invaderDistance': 10, 
            'stop': -100, 
            'reverse': 0}

  def findLastEatenFood(self, gameState):
    res = list()
    if len(self.observationHistory) > 1:
      previousState = self.getPreviousObservation()
      previousFood = self.getFoodYouAreDefending(previousState).asList()
      currentFood = self.getFoodYouAreDefending(gameState).asList()
      for food in previousFood:
        if food not in currentFood:
          res.append(food)
    return res

  def findEntrance(self, gameState):
    res = list()
    if self.red:
      boundary = {(gameState.data.layout.width/2-1, a) for a in range(gameState.data.layout.height)}
    else:
      boundary = {(gameState.data.layout.width/2, a) for a in range(gameState.data.layout.height)}
    walls = set(gameState.getWalls().asList())
    for position in boundary:
      if position not in walls:
        res.append(position)
    return res

  def getLegalActionsNoCrossing(self, gameState):
    actions = list()
    # print(gameState.getLegalActions(self.index))    
    for direction in gameState.getLegalActions(self.index):
      x, _ = gameState.getAgentPosition(self.index)
      dx, _ = Actions.directionToVector(direction)
      nextx = int(x + dx)
      if nextx <= self.maxX and nextx >= self.minX:
        actions.append(direction)
    print('legal actions:' + str(actions))
    return actions

  
###################
# Problem classes #
###################
class GoToProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState, defenderAgent, extra_walls=[]):
      self.start = (defenderAgent.myPosition, set(defenderAgent.target))
      self.walls = set(startingGameState.getWalls().asList()+extra_walls)
      self.startingGameState = startingGameState
      self.heuristicInfo = {} # A dictionary for the heuristic to store information
      self.maxX = defenderAgent.maxX
      self.minX = defenderAgent.minX

    def getStartState(self):
      return self.start

    def isGoalState(self, state):
      return state[0] in state[1]

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls and nextx <= self.maxX and nextx >= self.minX:
                nextTarget = set(state[1])
                successors.append( ( ((nextx, nexty), nextTarget), direction, 1) )
        return successors


class ChaseInvadersProblem2:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState, defenderAgent, extra_walls=[]):
      self.start = (defenderAgent.myPosition, set(defenderAgent.target))
      self.walls = set(startingGameState.getWalls().asList()+extra_walls)
      self.startingGameState = startingGameState
      self.heuristicInfo = {} # A dictionary for the heuristic to store information
      self.maxX = defenderAgent.maxX
      self.minX = defenderAgent.minX

    def getStartState(self):
      return self.start

    def isGoalState(self, state):
      return state[0] in state[1]

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextTarget = set(state[1])
                successors.append( ( ((nextx, nexty), nextTarget), direction, 1) )
        return successors


class GoToLastEatenFoodProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState, defenderAgent, extra_walls=[]):
      self.start = (defenderAgent.myPosition, set(defenderAgent.lastEatenFood))
      self.walls = set(startingGameState.getWalls().asList()+extra_walls)
      self.startingGameState = startingGameState
      self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
      return self.start

    def isGoalState(self, state):
      return state[0] in state[1]

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextTarget = set(state[1])
                successors.append( ( ((nextx, nexty), nextTarget), direction, 1) )
        return successors


class GoToEntranceProblem:
    def __init__(self, startingGameState, defenderAgent, extra_walls=[]):
      self.start = (defenderAgent.myPosition, set(defenderAgent.target))
      self.walls = set(startingGameState.getWalls().asList()+extra_walls)
      self.startingGameState = startingGameState
      self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
      return self.start

    def isGoalState(self, state):
      return state[0] in state[1]

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextTarget = set(state[1])
                successors.append( ( ((nextx, nexty), nextTarget), direction, 1) )
        return successors

class SearchSafeFoodProblem:
  def __init__(self, startingGameState, reaperAgent, enemyLocation, enemyWeight):
    self.start = (reaperAgent.myPosition, set(reaperAgent.safeFood))
    self.walls = set(startingGameState.getWalls().asList()) #set((walls))
    self.startingGameState = startingGameState
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
    self.agent = reaperAgent
    self.enemyLocation = enemyLocation
    self.enemyWeight = enemyWeight

  def getStartState(self):
    return self.start

  def isGoalState(self, state):
    return state[0] in self.start[1] #len(state[1]) == 0

  def getSuccessors(self, state):
      "Returns successor states, the actions they require, and a cost of 1."
      successors = []
      for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x,y = state[0]
          dx, dy = Actions.directionToVector(direction)
          nextx, nexty = int(x + dx), int(y + dy)
          #print(direction, nextx, nexty)
          if (nextx, nexty) not in self.walls and (not self.enemyLocation or\
            min(self.agent.distancer.getDistance((nextx, nexty), enemy) for enemy in self.enemyLocation)\
              > min([self.enemyWeight-1, self.agent.distancer.getDistance((nextx, nexty), self.start[0])])):
              nextFood = set(state[1])
              # if (nextx, nexty) in nextFood:
              #   nextFood.remove((nextx, nexty))
              successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
      return successors

class SearchDangerProblem_depth1(SearchSafeFoodProblem):
  def __init__(self, startingGameState, reaperAgent, enemyLocation, enemyWeight):
    self.start = (reaperAgent.myPosition, set(reaperAgent.safeFood+reaperAgent.dangerousFood_depth1))
    self.walls = set(startingGameState.getWalls().asList()) #set((walls))
    self.startingGameState = startingGameState
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
    self.agent = reaperAgent
    self.enemyLocation = enemyLocation
    self.enemyWeight = enemyWeight

class SearchDangerProblem_depth2(SearchSafeFoodProblem):
  def __init__(self, startingGameState, reaperAgent, enemyLocation, enemyWeight):
    self.start = (reaperAgent.myPosition, set(reaperAgent.safeFood+reaperAgent.dangerousFood_depth1+reaperAgent.dangerousFood_depth2))
    self.walls = set(startingGameState.getWalls().asList()) #set((walls))
    self.startingGameState = startingGameState
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
    self.agent = reaperAgent
    self.enemyLocation = enemyLocation
    self.enemyWeight = enemyWeight


class SearchNearestProblem(SearchSafeFoodProblem):
  def __init__(self, startingGameState, reaperAgent, enemyLocation, enemyWeight):
    self.start = (reaperAgent.myPosition, set(reaperAgent.food.asList()))
    self.walls = set(startingGameState.getWalls().asList()) #set((walls))
    self.startingGameState = startingGameState
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
    self.agent = reaperAgent
    self.enemyLocation = enemyLocation
    self.enemyWeight = enemyWeight

class GoHomeProblem(SearchSafeFoodProblem):
  def __init__(self, startingGameState, reaperAgent, enemyLocation, enemyWeight):
    self.start = (reaperAgent.myPosition, set(reaperAgent.homeBoundaryArea))
    print(self.start)
    self.walls = set(startingGameState.getWalls().asList()) #set((walls))
    self.startingGameState = startingGameState
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
    self.agent = reaperAgent
    self.enemyLocation = enemyLocation
    self.enemyWeight = enemyWeight




##############
# Heuristics # 
##############


def foodHeuristic1(state, problem, agent):
    
    position = state[0]
    foodGrid = set(state[1])
    if len(state) == 3 and state[2] == False:
        return manhattanHeuristic(position, problem)

    #gameState = problem.startingGameState

    if len(foodGrid) == 0:
        return 0
    elif len(foodGrid) == 1:
        return agent.distancer.getDistance(position, foodGrid.pop())

    min_d = 9999999
    for food in foodGrid:
      min_d = min(min_d, agent.distancer.getDistance(position, food))

    return min_d


def foodHeuristic2(state, problem, agent):
    
    position = state[0]
    foodGrid = set(state[1])
    if len(state) == 3 and state[2] == False:
        return manhattanHeuristic(position, problem)

    #gameState = problem.startingGameState

    if len(foodGrid) == 0:
        return 0
    elif len(foodGrid) == 1:
        return agent.distancer.getDistance(position, foodGrid.pop())

    if 'table' not in problem.heuristicInfo:
      problem.heuristicInfo['table'] = {}
    table = problem.heuristicInfo['table']

    cur_table = []
    for food1 in foodGrid:
      for food2 in foodGrid:
        pointwise = tuple(sorted([food1, food2]))
        if pointwise not in table:
          table[pointwise] = agent.distancer.getDistance(food1, food2)
        cur_table.append((pointwise, table[pointwise]))

    (p1, p2), farest = max(cur_table, key = lambda x:x[1])

    l1 = agent.distancer.getDistance(position, p1)
    l2 = agent.distancer.getDistance(position, p2)
    m = min(l1, l2) + farest
    return m

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5


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
  return (len(history)>3 and history[-4:-2]==history[-2:] and history[-1] != history[-2]) \
    or (len(history)>5 and history[-6:-3]==history[-3:] and history[-1] != history[-2])

def getSemiDangerousCells(safeCells, dangerousCells):
  dangerousFood_depth1 = []
  dangerousFood_depth2 = []
  for point in dangerousCells:
    if (point[0]+1, point[1]) in safeCells or (point[0], point[1]+1) in safeCells or \
      (point[0]-1, point[1]) in safeCells or (point[0], point[1]-1) in safeCells:
      dangerousFood_depth1.append(point)
  for point in dangerousCells:
    if point not in dangerousFood_depth1:
      if (point[0]+1, point[1]) in dangerousFood_depth1 or (point[0], point[1]+1) in dangerousFood_depth1 or \
        (point[0]-1, point[1]) in dangerousFood_depth1 or (point[0], point[1]-1) in dangerousFood_depth1:
        dangerousFood_depth2.append(point) 

  return dangerousFood_depth1, dangerousFood_depth2


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
    best_g = {}
    
    while not heap.isEmpty():
      node = heap.pop()
      g = node[1]
      node[0] = (node[0][0], tuple(node[0][1]))
      #print(node)
      if node[0] not in closed or g < best_g[node[0]]:
        closed[node[0]] = [g, node[2], node[3]] #state:[g,action,father]
        best_g[node[0]] = g

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


