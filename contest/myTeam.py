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
from util import manhattanDistance
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AlphaAgent', second = 'AlphaAgent'):
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
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class SuperKingPacAgent(CaptureAgent):

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
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

  def evaluationFunction(self, gameState):
    # Our plan:
    # Winning > Not getting killed > eating food > moving closer to food > fearing ghosts (see: God)
    ghostStates = [gameState.getGhostState(index) for index in self.getOpponents(gameState)]
    n = self.getFood(gameState).count()
    pos = gameState.getPacmanPosition()
    foodStates = self.getFood(gameState)
    capsules = self.getCapsules(gameState)

    # If you can win that's the best possible move
    if gameState.isWin():
        return 99999 + random.uniform(0, .5)

    if gameState.isLose():
        return -99999

    # Fear
    fear = 0
    fear_factor = 10
    ghosts = []
    gamma = .5

    # Calculate distances to nearest ghost
    if ghostStates:
        for ghost in ghostStates:
            if ghost.scaredTimer == 0:
                md = manhattanDistance(ghost.getPosition(), pos)
                ghosts.append(md)


    # Sort ghosts based on distance
    ghosts = sorted(ghosts)
    # Only worry about ghosts if they're nearby
    ghosts = [ghost for ghost in ghosts if ghost < 5]


    for i in range(len(ghosts)):
        # Fear is sum of the recipricals of the distances to the nearest ghosts multiplied
        # by a gamma^i where 0<gamma<1 and by a fear_factor
        fear += (fear_factor/ghosts[i]) * (gamma**i)

    # Record food coordinates
    foods = []
    for i in range(len(foodStates)):
        for j in range(len(foodStates[i])):
            if foodStates[i][j]:
                foods.append((i, j))

    #Calculate distances to nearest foods
    foodDistances = []
    if foods:
        for food in foods:
            md = manhattanDistance(food, pos)
            foodDistances.append(md)
    foodDistances = sorted(foodDistances)


    hunger_factor = 18
    # Hunger factor
    hunger = 0
    foodGamma = -0.4
    for i in range(len(foodDistances)):
        # Hunger is the sum of the reciprical of the distances to the nearest foods multiplied
        # by a foodGamma^i where 0<foodGamma<1 and by a hunger_factor
        hunger += (hunger_factor/foodDistances[i]) * (foodGamma**i)

    # Beserk mode
    scaredGhosts = []
    for ghost in ghostStates:
        if ghost.scaredTimer > 0:
            md = manhattanDistance(ghost.getPosition(), pos)
            scaredGhosts.append(md)

    # Senzu bean
    capsuleDistances = []
    for capsule in capsules:
      md = manhattanDistance(capsule, pos)
      capsuleDistances.append(md)

    capsuleDistances = sorted(capsuleDistances)
    for i in range(len(capsuleDistances)):
        hunger += (hunger_factor*4/capsuleDistances[i]) * (foodGamma**i)

    scaredGhosts = sorted(scaredGhosts)
    scaredGhosts = [ghost for ghost in scaredGhosts if ghost < 5]
    for i in range(len(scaredGhosts)):
        hunger += (hunger_factor*2/scaredGhosts[i]) * (foodGamma**i)

    score =  hunger - fear + random.uniform(0, .5) - (n+7)**2 + gameState.getScore() - (len(capsules)+30)**2
    return score
