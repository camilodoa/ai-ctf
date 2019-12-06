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

import random, time, util, game, itertools
from game import Directions, Actions
from captureAgents import CaptureAgent
from util import manhattanDistance

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='RationalAgent', second='RationalAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

# Global variables shared by the two agents
static_particles = None

aggressive_weights = {
    'closest-food' : -5.6628632625195814,
    'reverse' : -25.782246190694242,
    'closest-ghost' : 0.092345634713250316,
    'stop' : -30.264430656804493,
    'distance-to-friend' : 2.5003935434660369,
    'ghosts-1-step-away' : -100.9683911932976,
    'distance-from-home' : -10000.920029901729379,
    'opponents-4-steps-away' : -3.0550523826620437,
    'successor-food-count' : 100.0021893740633943063,
    'died' : -4000.5819301567078,
    'eats-food' : 10.622005393403555
}

defensive_weights = {
    'is-scared' : -106.6844053716024,
    'reverse' : -25.782246190694242,
    'closest-opp' : -202.05772051407411,
    'distance-to-start' : -50.5319388749071949,
    'stop' : -30.11940638920235,
    'eats-pacman' : 1000.184096316507649,
    'num-opps': -266.13745693758131,
    'closest-killer' : 0.4731431889471642
}


'''
SmartAgent #####################################################################
'''
class SmartAgent(CaptureAgent):
  '''
  General #####################################################################
  '''
  def registerInitialState(self, state):
      """
      This method handles the initial setup of the
      agent to populate useful fields (such as what team
      we're on).

      A distanceCalculator instance caches the maze distances
      between each pair of positions, so your agents can use:
      self.distancer.getDistance(p1, p2)

      IMPORTANT: This method may run for at most 15 seconds.
      """
      start = time.time()
      CaptureAgent.registerInitialState(self, state)

      # Start up particle filtering
      self.num_opponents = len(self.getOpponents(state))
      self.legal_positions = [p for p in state.getWalls().asList(False) if p[1] > 1]
      self.num_particles = 600
      self.initializeParticles(state)

      opp = self.getOpponents(state)
      walls = state.getWalls()

      self.start = state.getAgentPosition(self.index)
      self.border = walls.width / 2

      # Q learning
      self.epsilon = 0.0
      self.discount = 0.8
      self.alpha = 0.3
      self.reward = -1
      self.isOnRedTeam = state.isOnRedTeam(self.index)
      self.learn = True
      self.numOurFood = self.getFoodYouAreDefending(state).count(True)
      self.numFood = self.getFood(state).count(True)
      self.food = state.getBlueFood() if self.isOnRedTeam else state.getRedFood()

      if self.isOnRedTeam:
          self.home = (self.border - self.border/2, walls.height/2)
      else:
          self.home = (self.border + self.border/2, walls.height/2)

      if self.home not in self.legal_positions:
          self.home = min([(manhattanDistance(self.home, position), position) for position in self.legal_positions],key = lambda x : x[0])[1]

      # Minimax
      self.use_minimax = False
      self.depth = 4


  def chooseAction(self, state):
      start = time.time()

      self.debugDraw([(0,0)], [0,0,0], clear = True)
      self_agent = state.getAgentState(self.index)
      actions = state.getLegalActions(self.index)
      food = state.getBlueFood().count(True) if self.isOnRedTeam else state.getRedFood().count(True)

      # Particle filtering
      self.observeState()

      # If we're carrying enough, just go home!
      if self_agent.numCarrying >= 3:
        return self.returnHome(state)

      # Otherwise, run minimax
      elif self.use_minimax:
          startTime = start
          max_score = -99999
          max_action = None
          alpha = -99999
          beta = 99999
          for action in actions:
              if action != "Stop":
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

     # Or compute action from q-values
      else:
          action = random.choice(actions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

      # Q-learning
      if self.learn:
          reward = self.getReward(state.generateSuccessor(self.index, action), state)
          self.update(state, action, state.generateSuccessor(self.index, action), reward)

      # Draw particle distribution
      self.drawBeliefs()

      # Update particles
      self.elapseTime(state)

      end = time.time()
      if end - start > 1 : print("Overtime --> total time was ", end - start)

      return action


  def returnHome(self, state):
      '''
        Returns best action to get you home, avoiding opponents
      '''
      weights = {
        'distance-from-home' : -100.920029901729379,
        'closest-ghost' : 20.092345634713250316,
        'stop' : -30.264430656804493,
        'ghosts-1-step-away' : -2000.9683911932976,
        'opponents-4-steps-away' : -3.0550523826620437,
        'died' : -4000.5819301567078,
        'best-action' : 400
      }
      return None if len(state.getLegalActions(self.index)) == 0 else max([ (self.getHomeboundFeatures(state, action).__mul__(weights), action) for action in state.getLegalActions(self.index) ], key=lambda x : x[0])[1]

  def getHomeboundFeatures(self, state, action):
      features = util.Counter()
      x, y = pos = state.getAgentPosition(self.index)
      successor = state.generateSuccessor(self.index, action)
      actions = state.getLegalActions(self.index)
      # Meta data
      walls = state.getWalls()
      opponents = self.getLikelyOppPosition()
      ghosts = []
      oppPacmen = []

      # Fill out opponent arrays
      if self.isOnRedTeam:
          oppindices = state.getBlueTeamIndices()
          for i, opp in enumerate(opponents):
              if opp[0] < self.border:
                  oppPacmen.append(opp)
              else:
                  if state.getAgentState(oppindices[i]).scaredTimer == 0:
                      ghosts.append(opp)
      else:
          oppindices = state.getRedTeamIndices()
          for i, opp in enumerate(opponents):
              if opp[0] >= self.border:
                  oppPacmen.append(opp)
              else:
                  if state.getAgentState(oppindices[i]).scaredTimer == 0:
                      ghosts.append(opp)

      best_dist = 9999
      for spec_action in actions:
          successor = state.generateSuccessor(self.index, spec_action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.home, pos2)
          if dist < best_dist:
              best_action = spec_action
              best_dist = dist

      if best_action == action:
          features['best-action'] = 1

      next_x, next_y = self.generateSuccessorPosition(pos, action)

      # count the number of ghosts 1-step away
      ghostsOneStepAway = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

       # count the number of opponents that are 4 steps or fewer away
      oppFourStepsAway = sum(1 for ghost in ghosts if self.getMazeDistance((next_x, next_y), ghost) <= 4)

      # Only one feature if a ghost killed us
      if (next_x, next_y) in ghosts:
          features['died'] = 1.0
          features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
      # Only one feature if we're about to die
      elif ghostsOneStepAway >= 1:
          features['ghosts-1-step-away'] = float(ghostsOneStepAway) / len(ghosts)
          features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
      # Only one feature if there are opponents fewer than 4 steps away
      elif oppFourStepsAway >= 1:
          features['opponents-4-steps-away'] = float(oppFourStepsAway) / len(ghosts)
          features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)

      if len(ghosts) >= 1:
          dists = [self.dist((next_x, next_y), pac, walls) for pac in ghosts]
          features['closest-ghost'] = float(min(dists)) / (walls.width * walls.height)

      if action == Directions.STOP: features['stop'] = 1

      features.divideAll(10.0)
      return features


  def drawBeliefs(self):
      sorted_beliefs = self.getBeliefDistribution().sortedKeys()

      if len(sorted_beliefs) > 10:
          for i in range(10,0,-1):
            decimal = float(i)/10
            self.debugDraw([sorted_beliefs[i][0]], [decimal,0,decimal], clear = False)
            self.debugDraw([sorted_beliefs[i][1]], [0,decimal,decimal], clear = False)

      self.debugDraw([self.home], [1,0,0], clear = False)
  '''
  ###################################################################### General
  '''

  '''
  Expectimax ###################################################################
  '''
  def minimax(self, s, t, turn, alpha, beta, depth):
      if s.isOver():
          return self.evaluationFunction(s)

      if self.cutoffTest(t, 0.5, depth):
          return self.evaluationFunction(s)

      teams = self.getTeam(s) + self.getOpponents(s)

      if turn == 0 or turn == 1:
          max_action = -99999
          actions = s.getLegalActions(teams[turn])

          for action in actions:
              if action != "Stop":
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
              if action != "Stop":
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

  '''
  ################################################################### Expectimax
  '''

  '''
  Particle Filtering ###########################################################
  '''
  def initializeParticles(self, state):
      global static_particles
      if static_particles is None:
          permutations = list(itertools.product(self.legal_positions, repeat=self.num_opponents))
          random.shuffle(permutations)

          particlesPerPerm = self.num_particles // len(permutations)
          self.particles = []
          for permutation in permutations:
              for i in xrange(particlesPerPerm):
                  self.particles.append(permutation)

          remainderParticles = self.num_particles - (particlesPerPerm * len(permutations))
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
      state = self.getCurrentObservation()
      pacmanPosition = state.getAgentPosition(self.index)
      oppList = self.getOpponents(state)
      noisyDistances = [state.getAgentDistances()[x] for x in oppList]
      food = state.getRedFood() if self.isOnRedTeam else state.getBlueFood()
      if len(noisyDistances) < self.num_opponents:
          return

      #If we see a ghost, we know a ghost
      for i, opp in enumerate(oppList):
          pos = state.getAgentPosition(opp)
          if pos is not None:
              for j, particle in enumerate(self.particles):
                  newParticle = list(particle)
                  newParticle[i] = pos
                  self.particles[j] = tuple(newParticle)

          else:
              for j, particle in enumerate(self.particles):
                  distance = util.manhattanDistance(pacmanPosition, particle[i])
                  if distance <= 5:
                      newParticle = list(particle)
                      newParticle[i] = state.getInitialAgentPosition(opp)
                      self.particles[j] = tuple(newParticle)

      opp_position = self.getLikelyOppPosition()
      # If some of our food is missing, we know where a culprit might be!
      if food.count(True) != self.food.count(True):
          greedy_position = None
          for i, row in enumerate(food):
              for j, element in enumerate(row):
                  if food[i][j] != self.food[i][j]:
                      greedy_position = (i, j)
                      break

          self.food = food

          greedy_opp = min([(util.manhattanDistance(greedy_position, opp_position[i]), i) for i, opponent in enumerate(oppList)], key = lambda x : x[0])[1]
          for i, particle in enumerate(self.particles):
              newParticle = list(particle)
              newParticle[greedy_opp] = greedy_position
              self.particles[i] = tuple(newParticle)

      weights = util.Counter()
      for particle in self.particles:
          weight = 1
          for i, opponent in enumerate(oppList):
              distance = util.manhattanDistance(pacmanPosition, particle[i])
              # If we thought a ghost was near us but evidence is against it, prob is 0
              if state.getAgentPosition(opponent) is None and distance <= 5:
                  prob = 0
              else:
                  prob = state.getDistanceProb(distance, noisyDistances[i])
              weight *= prob
          weights[particle] += weight

      if weights.totalCount() == 0:
          self.particles = self.initializeParticles(state)
      else:
          weights.normalize()
          for i in xrange(len(self.particles)):
              self.particles[i] = util.sample(weights)
      static_particles = self.particles

  def elapseTime(self, state):
      global static_particles
      self.particles = static_particles
      newParticles = []
      for oldParticle in self.particles:
          newParticle = list(oldParticle)  # A list of ghost positions
          # Loop through and update each entry in newParticle...
          for i, opponent in enumerate(self.getOpponents(state)):
              currState = self.setGhostPosition(state, oldParticle[i], opponent)

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
      return beliefs

  def setGhostPosition(self, state, ghostPosition, oppIndex):
      "Sets the position of all ghosts to the values in ghostPositionTuple."
      conf = game.Configuration(ghostPosition, game.Directions.STOP)
      state.data.agentStates[oppIndex] = game.AgentState(conf, False)
      return state

  def setGhostPositions(self, state, ghostPositions, oppIndices):
      "Sets the position of all ghosts to the values in ghostPositionTuple."
      for index, pos in enumerate(ghostPositions):
          conf = game.Configuration(pos, game.Directions.STOP)
          state.data.agentStates[oppIndices[index]] = game.AgentState(conf, False)
      return state

  def getLikelyOppPosition(self):
      beliefs = self.getBeliefDistribution()
      return beliefs.argMax()
  '''
  ########################################################### Particle Filtering
  '''

  '''
  Q-Learning ###################################################################
  '''
  def update(self, state, action, nextState, reward):
    """
       Updates weights based on transition
    """
    difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
    for feature in self.getFeatures(state, action).sortedKeys():
        self.weights[feature] = self.weights[feature] + self.alpha*difference*self.getFeatures(state, action)[feature]

    return


  def computeActionFromQValues(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    # print([(self.getQValue(state, action), action, self.getFeatures(state, action)) for action in state.getLegalActions(self.index)])
    return None if len(state.getLegalActions(self.index)) == 0 else max([(self.getQValue(state, action), action) for action in state.getLegalActions(self.index)], key=lambda x : x[0])[1]

  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    return 0 if len(state.getLegalActions(self.index)) == 0 else max([self.getQValue(state, action) for action in state.getLegalActions(self.index)])

  def getWeights(self):
    util.raiseNotDefined()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator

      Includes time left to incetivize good moves early on
    """
    return self.getFeatures(state, action).__mul__(self.weights) + (state.data.timeleft - 1000)


  def closestFood(self, pos, food, walls):
        """
        Closest food from pos
        """
        fronteir = [(pos[0], pos[1], 0)]
        expanded = set()
        while fronteir:
            pos_x, pos_y, dist = fronteir.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fronteir.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

  def dist(self, pos, other_pos, walls):
        '''
        Closest distance from pos to other_pos
        '''
        fronteir = [(pos[0], pos[1], 0)]
        expanded = set()
        while fronteir:
            pos_x, pos_y, dist = fronteir.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if (pos_x, pos_y) == other_pos:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fronteir.append((nbr_x, nbr_y, dist+1))
        # no pac found
        return None

  def getFeatures(self, state, action):
        return util.raiseNotDefined()

  def evaluationFunction(self, state):
        return self.computeValueFromQValues(state)

  def getReward(self, state, lastState):
      return -1
  '''
  ################################################################### Q-Learning
  '''

'''
##################################################################### SmartAgent
'''

'''
AggressiveAgent ################################################################
'''
class AggressiveAgent(SmartAgent):
    def registerInitialState(self, state):
        SmartAgent.registerInitialState(self, state)
        self.weights = self.getWeights()

    def getFeatures(self, state, action):
        # Agressive features
        features = util.Counter()
        x, y = pos = state.getAgentPosition(self.index)
        successor = state.generateSuccessor(self.index, action)
        agentState = state.getAgentState(self.index)
        # Meta data
        walls = state.getWalls()
        opponents = self.getLikelyOppPosition()
        ghosts = []
        oppPacmen = []

        # Fill out opponent arrays
        if self.isOnRedTeam:
            oppindices = state.getBlueTeamIndices()
            food = state.getBlueFood()
            for i, opp in enumerate(opponents):
                if opp[0] < self.border:
                    oppPacmen.append(opp)
                else:
                    if state.getAgentState(oppindices[i]).scaredTimer == 0:
                        ghosts.append(opp)
            friendPos = state.getAgentPosition([x for x in state.getRedTeamIndices() if x != self.index][0])
            if pos[0] > self.border and friendPos[0] > self.border:
                bothOffside = True
            else:
                bothOffside = False
        else:
            food = state.getRedFood()
            oppindices = state.getRedTeamIndices()
            for i, opp in enumerate(opponents):
                if opp[0] >= self.border:
                    oppPacmen.append(opp)
                else:
                    if state.getAgentState(oppindices[i]).scaredTimer == 0:
                        ghosts.append(opp)
            friendPos = state.getAgentPosition([x for x in state.getBlueTeamIndices() if x != self.index][0])
            if pos[0] <= self.border and friendPos[0] <= self.border:
                bothOffside = True
            else:
                bothOffside = False

        next_x, next_y = self.generateSuccessorPosition(pos, action)

        # count the number of ghosts 1-step away
        ghostsOneStepAway = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

         # count the number of opponents that are 4 steps or fewer away
        oppFourStepsAway = sum(1 for ghost in ghosts if self.getMazeDistance((next_x, next_y), ghost) <= 4)

        # Only one feature if a ghost killed us
        if (next_x, next_y) in ghosts:
            features['died'] = 1.0
            features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
        # Only one feature if we're about to die
        elif ghostsOneStepAway >= 1:
            features['ghosts-1-step-away'] = float(ghostsOneStepAway) / len(ghosts)
            features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
        # Only one feature if there are opponents fewer than 4 steps away
        elif oppFourStepsAway >= 1:
            features['opponents-4-steps-away'] = float(oppFourStepsAway) / len(ghosts)
            features['distance-from-home'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
        # Otherwise, we have regular features
        else:
            features['successor-food-count'] = -food.count(True)
            if food[next_x][next_y]:
                features['eats-food'] = 1.0

            if bothOffside:
                features['distance-to-friend'] = float(self.getMazeDistance(pos, friendPos)) / (walls.width * walls.height)

            dist = self.closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features['closest-food'] = float(dist) / (walls.width * walls.height)

            if len(ghosts) >= 1:
                dists = [self.dist((next_x, next_y), pac, walls) for pac in ghosts]
                features['closest-ghost'] = float(min(dists)) / (walls.width * walls.height)

            if action == Directions.STOP: features['stop'] = 1
            rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
            if action == rev: features['reverse'] = 1.0

        features.divideAll(10.0)
        return features


    def getReward(self, state, lastState):
        # Aggro reward only depends on food left
        return -state.getBlueFood().count(True) if self.isOnRedTeam else -state.getRedFood().count(True)

    def getWeights(self):
        global aggressive_weights
        return aggressive_weights
'''
################################################################ AggressiveAgent
'''

'''
DefensiveAgent #################################################################
'''
class DefensiveAgent(SmartAgent):
    def registerInitialState(self, state):
        SmartAgent.registerInitialState(self, state)
        self.weights = self.getWeights()

    def getFeatures(self, state, action):
        # Defensive features
        features = util.Counter()
        x, y = pos = state.getAgentPosition(self.index)
        successor = state.generateSuccessor(self.index, action)
        agentState = state.getAgentState(self.index)
        walls = state.getWalls()
        # Meta data

        walls = state.getWalls()
        opponents = self.getLikelyOppPosition()
        ghosts = []
        oppPacmen = []
        # Fill out opponent arrays
        if self.isOnRedTeam:
            if pos[0] > self.border:
                isOnside = False
            else:
                isOnside = True

            food = state.getRedFood()
            for opp in opponents:
                if opp[0] < self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)
        else:
            if pos[0] <= self.border:
                isOnside = False
            else:
                isOnside = True

            food = state.getBlueFood()
            for opp in opponents:
                if opp[0] >= self.border:
                    oppPacmen.append(opp)
                else:
                    ghosts.append(opp)

        next_x, next_y = self.generateSuccessorPosition(pos, action)

        dists = [self.dist((next_x, next_y), pac, walls) for pac in oppPacmen]

        if not isOnside:
            features['distance-to-start'] = float(self.getMazeDistance((next_x, next_y), self.start)) / (walls.width * walls.height)
        if len(oppPacmen) > 0:
            dists = [self.dist((next_x, next_y), pac, walls) for pac in oppPacmen]
            # if agentState.scaredTimer > 0:
            #     features['is-scared'] = 1
            #     features['closest-killer'] = float(min(dists)) / (walls.width * walls.height)
            features['num-opps'] = len(oppPacmen)
            features['closest-opp'] = float(min(dists)) / (walls.width * walls.height)
            if (next_x, next_y) in oppPacmen:
                features['eats-pacman'] = 1.0

        if action == Directions.STOP: features['stop'] = 1.0
        rev = Directions.REVERSE[state.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1.0

        features.divideAll(10.0)
        return features

    def getReward(self, state, lastState):
        # Defensive reward only depends on our food
        our_food = state.getRedFood().count(True) if self.isOnRedTeam else state.getBlueFood().count(True)

        return our_food - self.numOurFood # Goes from 0 to -numFood

    def getWeights(self):
        global defensive_weights
        return defensive_weights
'''
################################################################# DefensiveAgent
'''

'''
RationalAgent #################################################################
'''
class RationalAgent(DefensiveAgent, AggressiveAgent, SmartAgent):
    '''
    Idea behind this class:
    Whenever our agents are at a starting position, or if we're missing half of
    our food they get to pick to be a defensive agent or an offensive agent.
    - if there is an invader on our side
        - switch to defensive
    - otherwise
        - be aggressive!
    '''
    def registerInitialState(self, s):
        SmartAgent.registerInitialState(self, s)
        self.depth = 4

        global aggressive_weights
        self.weights = aggressive_weights

    def chooseAction(self, s):
        x, y = pos= s.getAgentPosition(self.index)
        opponents = self.getLikelyOppPosition()
        defenders = []
        invaders = []
        us = s.getAgentState(self.index)
        score = self.getScore(s)
        # Fill out opponent arrays
        if self.isOnRedTeam:
            for opp in opponents:
                if opp[0] < self.border:
                    invaders.append(opp)
                else:
                    defenders.append(opp)
        else:
            for opp in opponents:
                if opp[0] >= self.border:
                    invaders.append(opp)
                else:
                    defenders.append(opp)

        global defensive_weights
        global aggressive_weights
        defensive = (len(invaders) >= 1 and us.scaredTimer == 0) or (self.getFoodYouAreDefending(s).count(True) <= (self.numOurFood//2) or (score >= 9 and self.weights == defensive_weights))
        # If we're being invaded and we aren't scared, be defensive
        if defensive:
            if len(defenders) >= 2 and self.getMazeDistance(pos, self.home) > 9:
                action = self.returnHome(s)
                print("country roads... take me home")
            else:
                self.weights = defensive_weights
                action = DefensiveAgent.chooseAction(self, s)

        # Otherwise, be aggressive!
        else:
            self.weights = aggressive_weights
            action = AggressiveAgent.chooseAction(self, s)
        return action

    def getFeatures(self, s, a):
        global defensive_weights
        global aggressive_weights
        if self.weights == defensive_weights : return DefensiveAgent.getFeatures(self, s, a)
        elif self.weights == aggressive_weights : return AggressiveAgent.getFeatures(self, s, a)


'''
#################################################################  RationalAgent
'''
