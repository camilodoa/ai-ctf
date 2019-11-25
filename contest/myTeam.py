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
import pickle

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='GoodAggroAgent', second='GoodDefensiveAgent'):
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

static_particles = None

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
        self.depth = 10
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

            result = self.minimax(successor, startTime, 1, alpha, beta, 1)
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

    def minimax(self, s, t, turn, alpha, beta, depth):
        '''
        s: gameState
        t: time
        turn: 0 if pacman, 1 if ghost
        '''
        if s.isOver():
            return self.evaluationFunction(s)

        if self.cutoffTest(t, 0.6, depth):
            return self.evaluationFunction(s)

        teams = self.getTeam(s) + self.getOpponents(s)

        if turn == 0 or turn == 1:
            max_action = -99999
            actions = s.getLegalActions(teams[turn])

            for action in actions:
                result = self.minimax(s.generateSuccessor(teams[turn], action), t, turn + 1, alpha, beta, depth + 1)
                if result > max_action:
                    max_action = result

                if max_action > beta:
                    return max_action

                alpha = max(alpha, max_action)

            return max_action

        if turn >= 2:
            min_action = 99999
            actions = s.getLegalActions(teams[turn])

            for action in actions:
                if turn == 3:
                    newPos = self.generateSuccessorPosition(s.getAgentPosition(teams[turn]), action)
                    successor = self.setGhostPosition(s, newPos, teams[turn])
                    result = self.minimax(sucessor, t, turn + 1, alpha, beta, depth + 1)
                else:
                    newPos = self.generateSuccessorPosition(s.getAgentPosition(teams[turn]), action)
                    successor = self.setGhostPosition(s, newPos, teams[turn])
                    result = self.minimax(successor, t, 0, alpha, beta, depth + 1)

                if result < min_action:
                    min_action = result

                if min_action < alpha:
                    return min_action

                beta = min(beta, min_action)

            return min_action

    def cutoffTest(self, t, limit, depth):
        timeElapsed = time.time() - t
        if timeElapsed >= limit:
            return True
        elif depth >= self.depth:
            return True
        return False

    def evaluationFunction(self, gameState):
        return util.raiseNotDefined()

    '''
    ################################################################

                Particle Filtering

    ################################################################
    '''

    def initializeParticles(self, gameState):
        global static_particles
        if static_particles is None:
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
            static_particles = self.particles
            return self.particles
        else:
            self.particles = static_particles
            return self.particles

    def observeState(self):
        global static_particles
        self.particles = static_particles
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
        static_particles = self.particles

    def elapseTime(self, gameState):
        global static_particles
        self.particles = static_particles
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
        static_particles = self.particles

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
      opp = self.getOpponents(gameState)
      walls = gameState.getWalls()
      self.border =abs(self.start[0] - walls.width)/2

      self.selfHome = self.border + (self.start[0] - self.border)
      self.oppHome = self.border + (gameState.getInitialAgentPosition(opp[0])[0] - self.border)
      self.qValues = util.Counter()
      self.epsilon=0.1
      self.gamma = self.discount =0.8
      self.alpha=0.3
      self.reward = -0.1
      self.isOnRedTeam = gameState.isOnRedTeam(self.index)
      self.useMinimax = True
      self.save = True
      self.numOurFood = self.getFoodYouAreDefending(gameState).count(True)
      self.numFood = self.getFood(gameState).count(True)


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
      self.observeState()
      self.elapseTime(state)

      actions = state.getLegalActions(self.index)
      if self.useMinimax:
          if util.flipCoin(self.epsilon):
              action = random.choice(actions)
          else:
              self.computeActionFromQValues(state)

              startTime = time.time()
              max_score = -99999
              max_action = None
              alpha = -99999
              beta = 99999
              for action in actions:
                  # Update successor ghost positions to be the max pos in our particle distributions
                  successor = state.generateSuccessor(self.index, action)
                  ghosts = self.getBeliefDistribution().argMax()
                  successor = self.setGhostPositions(successor, ghosts, self.getOpponents(state))

                  result = self.minimax(successor, startTime, 1, alpha, beta, 1)
                  if result >= max_score:
                      max_score = result
                      max_action = action

                  if max_score > beta:
                      return max_action

                  alpha = max(alpha, max_score)

              action = max_action

          reward = self.getReward(state.generateSuccessor(self.index, action), state)
          self.update(state, action, state.generateSuccessor(self.index, action), reward)

      else:
          # Pick Action
          legalActions = state.getLegalActions(self.index)
          action = random.choice(legalActions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)
          reward = self.getReward(state.generateSuccessor(self.index, action), state)
          self.update(state, action, state.generateSuccessor(self.index, action), reward)

      if self.save:
          print("agent ", self.index, "'s weights are ", self.weights)
          file = open(self.weightfile,'wb')
          pickle.dump(self.weights, file)
          file.close()

      return action

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
        return util.raiseNotDefined()

  def evaluationFunction(self, state):
        return self.computeValueFromQValues(state)

  def getReward(self, state, lastState):
      food = state.getBlueFood().count(True) if self.isOnRedTeam else state.getRedFood().count(True)
      ourFood = state.getRedFood().count(True) if self.isOnRedTeam else state.getBlueFood().count(True)

      ourFood_score = self.numOurFood - ourFood  # Total minus current
      food_score = self.numFood - food # Total minus current

      opponents = self.getLikelyOppPosition()
      pos = state.getAgentPosition(self.index)
      ghosts = []
      oppPacmen = []
      # Fill out opponent arrays
      if self.isOnRedTeam:
          for opp in opponents:
              if opp[0] < self.border:
                  oppPacmen.append(opp)
              else:
                  ghosts.append(opp)
      else:
          for opp in opponents:
              if opp[0] >= self.border:
                  oppPacmen.append(opp)
              else:
                  ghosts.append(opp)

      # State score
      score = self.reward + state.getScore() - lastState.getScore()

      # Food score
      score += food_score - ourFood_score # Good score - loss
      print(ourFood_score, food_score)

      # Onside calculation
      if self.isOnRedTeam:
          if pos[0] > self.border:
              isOnside = False
          else:
              isOnside = True
      else:
          if pos[0] <= self.border:
              isOnside = False
          else:
              isOnside = True

      # isWin
      if food <= 2:
          if isOnside == True:
              score += 99999
              print("WIN!!")

      # isLose
      if ourFood <= 2:
          if len(oppPacmen) == 0:
              score -= 99999
              print("lose :-(")

      return score


class GoodAggroAgent(PacmanQAgent):
    def registerInitialState(self, gameState):
        BigBrainAgent.registerInitialState(self, gameState)
        PacmanQAgent.registerInitialState(self, gameState)
        self.epsilon = 0.3
        self.gamma = self.discount = 0.8
        self.alpha = 0.2
        self.reward = -1
        self.depth = 4
        self.useMinimax = False

        self.weightfile = "./GoodWeights1.pkl"
        self.weights = util.Counter()
        # file = open(self.weightfile, 'r')
        # self.weights = pickle.load(file)

    def getFeatures(self, state, action):
        # Agressive features
        features = util.Counter()
        pos = state.getAgentPosition(self.index)
        successor = state.generateSuccessor(self.index, action)
        # Meta data
        food = self.getFood(state)
        walls = state.getWalls()
        opponents = self.getLikelyOppPosition()
        ghosts = []
        oppPacmen = []
        # Fill out opponent arrays
        if self.isOnRedTeam:
            for opp in opponents:
                if opp[0] < self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)
        else:
            for opp in opponents:
                if opp[0] >= self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)

        features['onOffence'] = 1
        if not successor.getAgentState(self.index).isPacman: features['onOffence'] = 0

        features['successorNumFood'] = self.getFood(successor).count(True)

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

    def getReward(self, state, lastState):
        # Aggro reward only depends on food left
        pos = state.getAgentPosition(self.index)
        food = state.getBlueFood().count(True) if self.isOnRedTeam else state.getRedFood().count(True)
        food_score = self.numFood - food # Total minus current

        # State score
        score = self.reward + state.getScore() - lastState.getScore()

        # Food score
        score += food_score

        # Onside calculation
        if self.isOnRedTeam:
            if pos[0] > self.border:
                isOnside = False
            else:
                isOnside = True
        else:
            if pos[0] <= self.border:
                isOnside = False
            else:
                isOnside = True

        # isWin
        if food <= 2:
            if isOnside == True:
                score += 99999
                print("WIN!!")

        return score



class GoodDefensiveAgent(PacmanQAgent):
    def registerInitialState(self, gameState):
        BigBrainAgent.registerInitialState(self, gameState)
        PacmanQAgent.registerInitialState(self, gameState)
        self.epsilon = 0.3
        self.gamma = self.discount = 0.8
        self.alpha = 0.2
        self.reward = -1
        self.depth = 3
        self.useMinimax = False

        self.weightfile = "./GoodWeights2.pkl"
        self.weights = util.Counter()
        # file = open(self.weightfile, 'r')
        # self.weights = pickle.load(file)

    def getFeatures(self, state, action):
        # Defensive features
        features = util.Counter()
        pos = state.getAgentPosition(self.index)
        successor = state.generateSuccessor(self.index, action)
        # Meta data
        food = self.getFoodYouAreDefending(state)
        walls = state.getWalls()
        opponents = self.getLikelyOppPosition()
        ghosts = []
        oppPacmen = []
        # Fill out opponent arrays
        if self.isOnRedTeam:
            for opp in opponents:
                if opp[0] < self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)
        else:
            for opp in opponents:
                if opp[0] >= self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)


        features['numPac'] = len(oppPacmen)
        if len(oppPacmen) > 0:
          dists = [self.getMazeDistance(pos, a) for a in oppPacmen]
          features['closestPac'] = float(min(dists)) / (walls.width * walls.height)

        features['onDefense'] = 1.0
        if successor.getAgentState(self.index).isPacman: features['onDefense'] = 0

        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        if (next_x, next_y) in oppPacmen:
            features["eats-pacman"] = 1.0

        # Distance to start
        features["distance-to-start"] = float(self.getMazeDistance(pos, self.start)) / (walls.width * walls.height)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features.divideAll(10.0)
        return features

    def getReward(self, state, lastState):
        # Defensive reward only depends on our food
        ourFood = state.getRedFood().count(True) if self.isOnRedTeam else state.getBlueFood().count(True)

        ourFood_score = ourFood - self.numOurFood # Total minus current

        opponents = self.getLikelyOppPosition()
        ghosts = []
        oppPacmen = []
        # Fill out opponent arrays
        if self.isOnRedTeam:
            for opp in opponents:
                if opp[0] < self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)
        else:
            for opp in opponents:
                if opp[0] >= self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)

        # State score
        score = self.reward + state.getScore() - lastState.getScore()

        # Food score
        score += ourFood_score

        # isLose
        if ourFood <= 2:
            if len(oppPacmen) == 0:
                score -= 99999
                print("lose :-(")

        return score
