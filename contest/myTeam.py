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
from keyboardAgents import KeyboardAgent
import random, time, util
from util import manhattanDistance
from game import Directions
import game
from game import Directions, Actions
import itertools
import time

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='PacmanQAgent', second='PacmanQAgent'):
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

class BigBrainAgent(CaptureAgent):

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

        # Start up particle filtering
        self.numGhosts = len(self.getOpponents(gameState))
        self.ghostAgents = []
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.numParticles = 600
        self.initializeParticles(gameState)

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        startTime = time.time()
        actions = gameState.getLegalActions(self.index)

        # Particle filtering
        self.observeState()
        self.elapseTime(gameState)

        max_score = -99999
        max_action = None
        alpha = -99999
        beta = 99999
        for action in actions:
            # Update successor ghost positions to be the max pos in our particle distributions
            successor = gameState.generateSuccessor(self.index, action)
            ghosts = self.getBeliefDistribution().argMax()
            successor = self.setGhostPositions(successor, ghosts, self.getOpponents(gameState))

            result = self.minimax(successor, startTime, 1, alpha, beta)
            if result >= max_score:
                max_score = result
                max_action = action

            if max_score > beta:
                return max_action

            alpha = max(alpha, max_score)

        return max_action

    '''
     ################################################################

				Expectimax

    ################################################################
    '''

    # @memoize
    def minimax(self, s, t, turn, alpha, beta):
        '''
        s: gameState
        t: time
        turn: 0 if pacman, 1 if ghost
        '''
        if s.isOver():
            return self.evaluationFunction(s)

        if self.cutoffTest(t, 0.9):
            return self.evaluationFunction(s)

        teams = self.getTeam(s) + self.getOpponents(s)

        if turn == 0 or turn == 1:
            max_action = -99999
            actions = s.getLegalActions(teams[turn])

            for action in actions:
                result = self.minimax(s.generateSuccessor(teams[turn], action), t, turn + 1, alpha, beta)
                if result > max_action:
                    max_action = result

                if max_action > beta:
                    return max_action

                alpha = max(alpha, max_action)

            return max_action

        if turn >= 2:
            return 0
            min_action = 99999
            actions = s.getLegalActions(teams[turn])
            # Change this to setGhostPosition for highest probabilities

            for action in actions:
                if turn == 3:
                    result = self.minimax(s.generateSuccessor(teams[turn], action), t, 0, alpha, beta)
                else:
                    result = self.minimax(s.generateSuccessor(teams[turn], action), t, turn + 1, alpha, beta)

                if result < min_action:
                    min_action = result

                if min_action < alpha:
                    return min_action

                beta = min(beta, min_action)

            return min_action

    def cutoffTest(self, t, limit):
        timeElapsed = time.time() - t
        if timeElapsed > limit:
            return True
        return False

    def evaluationFunction(self, gameState):
        # Our plan:
        # Winning > Not getting killed > eating food > moving closer to food > fearing ghosts (see: God)
        ghostStates = [gameState.getAgentState(index) for index in self.getOpponents(gameState)]
        n = self.getFood(gameState).count()
        pos = gameState.getAgentPosition(self.index)
        foodStates = self.getFood(gameState)
        capsules = self.getCapsules(gameState)

        # # If you can win that's the best possible move
        # if gameState.isWin():
        #     return 99999 + random.uniform(0, .5)
        #
        # if gameState.isLose():
        #     return -99999

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
            fear += (fear_factor / ghosts[i]) * (gamma ** i)

        # Record food coordinates
        foods = []
        for i in range(len(foodStates)):
            for j in range(len(foodStates[i])):
                if foodStates[i][j]:
                    foods.append((i, j))

        # Calculate distances to nearest foods
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
            hunger += (hunger_factor / foodDistances[i]) * (foodGamma ** i)

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
            hunger += (hunger_factor * 4 / capsuleDistances[i]) * (foodGamma ** i)

        scaredGhosts = sorted(scaredGhosts)
        scaredGhosts = [ghost for ghost in scaredGhosts if ghost < 5]
        for i in range(len(scaredGhosts)):
            hunger += (hunger_factor * 2 / scaredGhosts[i]) * (foodGamma ** i)

        score = hunger - fear + random.uniform(0, .5) - (n + 7) ** 2 + gameState.getScore() - (len(capsules) + 30) ** 2
        return score

    '''
    ################################################################

                Particle Filtering

    ################################################################
    '''

    def initializeParticles(self, gameState):
        permutations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        random.shuffle(permutations)

        particlesPerPerm = self.numParticles // len(permutations)
        self.particles = []
        for permutation in permutations:
            for i in xrange(particlesPerPerm):
                self.particles.append(permutation)

        remainderParticles = self.numParticles - (particlesPerPerm * len(permutations))
        for i in xrange(remainderParticles):
            self.particles.append(random.choice(permutations))

        return self.particles

    def observeState(self):
        gameState = self.getCurrentObservation()
        pacmanPosition = gameState.getAgentPosition(self.index)
        oppList = self.getOpponents(gameState)
        noisyDistances = [gameState.getAgentDistances()[x] for x in oppList]
        if len(noisyDistances) < self.numGhosts:
            return

        #If we see a ghost, we know a ghost
        for i, opp in enumerate(oppList):
            pos = gameState.getAgentPosition(opp)
            if pos is not None:
                for j, particle in enumerate(self.particles):
                    newParticle = list(particle)
                    newParticle[i] = pos
                    self.particles[j] = tuple(newParticle)

        weights = util.Counter()
        for particle in self.particles:
            weight = 1
            for i, opponent in enumerate(oppList):
                distance = util.manhattanDistance(pacmanPosition, particle[i])
                prob = gameState.getDistanceProb(distance, noisyDistances[i])
                weight *= prob
            weights[particle] += weight

        if weights.totalCount() == 0:
            self.particles = self.initializeParticles(gameState)
        else:
            weights.normalize()
            for i in xrange(len(self.particles)):
                self.particles[i] = util.sample(weights)

    def elapseTime(self, gameState):
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in newParticle...

            for i, opponent in enumerate(self.getOpponents(gameState)):
                currState = self.setGhostPosition(gameState, oldParticle[i], opponent)

                actions = currState.getLegalActions(opponent)
                action = random.sample(actions, 1)[0]
                newParticle[i] = self.generateSuccessorPosition(oldParticle[i], action)

            newParticles.append(tuple(newParticle))
        self.particles = newParticles

    def generateSuccessorPosition(self, position, action):
        newPosition = list(position)
        if action == "West":
            newPosition[0] -= 1
        elif action == "East":
            newPosition[0] += 1
        elif action == "South":
            newPosition[1] -= 1
        elif action == "North":
            newPosition[1] += 1

        return tuple(newPosition)

    def getBeliefDistribution(self):
        beliefs = util.Counter()
        for particle in self.particles:
            beliefs[particle] += 1
        beliefs.normalize()
        self.debugDraw([beliefs.argMax()[0], beliefs.argMax()[1]], [0,.5,.5], clear = True)
        return beliefs

    def setGhostPosition(self, gameState, ghostPosition, oppIndex):
        "Sets the position of all ghosts to the values in ghostPositionTuple."
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[oppIndex] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions, oppIndices):
        "Sets the position of all ghosts to the values in ghostPositionTuple."
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[oppIndices[index]] = game.AgentState(conf, False)
        return gameState

    def getLikelyOppPosition(self):
        beliefs = self.getBeliefDistribution()
        return beliefs.argMax()


class PacmanQAgent(BigBrainAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
      BigBrainAgent.registerInitialState(self, gameState)

      self.start = gameState.getAgentPosition(self.index)
      self.qValues = util.Counter()
      self.epsilon=0.05
      self.gamma = self.discount =0.8
      self.alpha=0.3
      self.weights = util.Counter()


  def chooseAction(self, state):
      """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
      actions = state.getLegalActions(self.index)
      startTime = time.time()


      self.observeState()
      self.elapseTime(state)

      max_score = -99999
      max_action = None
      alpha = -99999
      beta = 99999
      for action in actions:
          # Update successor ghost positions to be the max pos in our particle distributions
          successor = state.generateSuccessor(self.index, action)
          ghosts = self.getBeliefDistribution().argMax()
          successor = self.setGhostPositions(successor, ghosts, self.getOpponents(state))

          result = self.minimax(successor, startTime, 1, alpha, beta)
          if result >= max_score:
              max_score = result
              max_action = action

          if max_score > beta:
              return max_action

          alpha = max(alpha, max_score)

      self.update(state, max_action, state.generateSuccessor(self.index, max_action), -10)
      # print(max_action, self.computeValueFromQValues(state.generateSuccessor(self.index, max_action)))
      return max_action

      # Pick Action
      # legalActions = state.getLegalActions(self.index)
      # action = random.choice(legalActions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)
      #
      # print(action, self.computeValueFromQValues(state))
      # return action

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)

    for feature in self.getFeatures(state, action).sortedKeys():
        self.weights[feature] += self.alpha*difference*self.getFeatures(state, action)[feature]

    return


  def computeActionFromQValues(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    return None if len(state.getLegalActions(self.index)) == 0 else max([ (self.getQValue(state, action), action) for action in state.getLegalActions(self.index)], key=lambda x : x[0])[1]

  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    return 0 if len(state.getLegalActions(self.index)) == 0 else max([self.getQValue(state, action) for action in state.getLegalActions(self.index)])

  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    return self.getFeatures(state, action).__mul__(self.weights)


  def closestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

  def getFeatures(self, state, action):
        '''
        # ADD TO FEATURES:
        --> Whether an opp is scared
        --> Whether an opp is pacman
        --> Number of our food left
        --> Number of their food left
        --> Our capsules / their capsules
        --> Are we scared
        --> How close are we to our teammate
        --> How close the ghosts are to each other
        --> score :-(
        '''

        # extract the grid of food and wall locations and get the ghost locations
        food = self.getFood(state)
        walls = state.getWalls()
        ghosts = self.getLikelyOppPosition()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

  def evaluationFunction(state):
        return self.computeValueFromQValues(state)





'''
DON'T LOOK DOWN HERE
'''
def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """

    class memodict(dict):
        def __init__(self, func):
            print(dir(func))
            self.func = func

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            result = self[key] = self.func(*key)
            return result

    return memodict(f)
