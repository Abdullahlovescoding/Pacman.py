# main.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
Pacman.py holds the logic for the classic pacman game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the pacman world:
          Pacman is a complex environment.  You probably don't want to
          read through all of the code we wrote to make the game runs
          correctly.  This section contains the parts of the code
          that you will need to understand in order to complete the
          project.  There is also some code in game.py that you should
          understand.

  (ii)  The hidden secrets of pacman:
          This section contains all of the logic code that the pacman
          environment uses to decide who can move where, who dies when
          things collide, etc.  You shouldn't need to read this section
          of code, but you can if you want.

  (iii) Framework to start a game:
          The final section contains the code for reading the command
          you use to set up the game, then starting up a new game, along with
          linking in all the external parts (agent functions, graphics).
          Check this section out to see all the options available to you.

To play your first game, type 'python pacman.py' from the command line.
The keys are 'a', 's', 'd', and 'w' to move (or arrow keys).  Have fun!
"""
import os
import random
import sys
import layout
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint, manhattanDistance


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.

    Note that in classic Pacman, Pacman is always agent 0.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    explored = set()

    def get_and_reset_explored():
        """
        :return:
        """
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    get_and_reset_explored = staticmethod(get_and_reset_explored)

    def get_legal_actions(self, agent_index=0):
        """
        Returns the legal actions for the agent specified.
        """
        GameState.explored.add(self)
        if self.is_win() or \
                self.isLose():
            return []

        if agent_index == 0:  # Pacman is moving
            return PacmanRules.get_legal_actions(self)
        return GhostRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.is_win() or self.isLose():
            raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agent_index == 0:
            state.data._eaten = [False for i in range(state.getNumAgents())]
            PacmanRules.applyAction(state, action)
        else:  # A ghost is moving
            GhostRules.applyAction(state, action, agent_index)

        # Time passes
        if agent_index == 0:
            state.data.scoreChange += -TIME_PENALTY  # Penalty for waiting around
        else:
            GhostRules.decrement_timer(state.data.agentStates[agent_index])

        GhostRules.checkDeath(state, agent_index)

        state.data._agentMoved = agent_index
        state.data.score += state.data.scoreChange
        return state

    def get_legal_pacman_actions(self):
        return self.get_legal_actions(0)

    def generate_pacman_successor(self, action):
        """
        Generates the successor state after the specified pacman move
        """
        return self.generate_successor(0, action)

    def get_pacman_state(self):
        """
        Returns an AgentState object for pacman (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.data.agentStates[0].copy()

    def getPacmanPosition(self):
        return self.data.agentStates[0].getPosition()

    def getGhostStates(self):
        return self.data.agentStates[1:]

    def getGhostState(self, agent_index):
        if agent_index == 0 or agent_index >= self.getNumAgents():
            raise Exception("Invalid index passed to getGhostState")
        return self.data.agentStates[agent_index]

    def getGhostPosition(self, agent_index):
        if agent_index == 0:
            raise Exception("Pacman's index passed to getGhostPosition")
        return self.data.agentStates[agent_index].getPosition()

    def getGhostPositions(self):
        return [s.getPosition() for s in self.getGhostStates()]

    def getNumAgents(self):
        return len(self.data.agentStates)

    def get_score(self):
        return self.data.score

    def get_capsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        """
        return self.data.capsules

    def getNumFood(self):
        return self.data.food.count()

    def get_food(self):
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.get_food()
        if currentFood[x][y] == True: ...
        """
        return self.data.food

    def get_walls(self):
        """
        Returns a Grid of boolean wall indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        walls = state.get_walls()
        if walls[x][y] == True: ...
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        return self.data.food[x][y]

    def hasWall(self, x, y):
        return self.data.layout.walls[x][y]

    def isLose(self):
        return self.data.lose

    def is_win(self):
        return self.data.win

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prevState=None):
        """
        Generates a new state by copying information from its predecessor.
        """
        if prevState is not None:  # Initial state
            self.data = GameStateData(prevState.data)
        else:
            self.data = GameStateData()

    def deepCopy(self):
        state = GameState(self)
        state.data = self.data.deepCopy()
        return state

    def __eq__(self, other=None):
        """
        Allows two states to be compared.
        """
        # print("Printing data")
        # print(self.data)
        # print("Printing other")
        # print(other)
        if not other:
            return None

        return self.data == other.data

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        return hash(self.data)

    def __str__(self):

        return str(self.data)

    def initialize(self, layout, numGhostAgents=1000):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.data.initialize(layout, numGhostAgents)


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

SCARED_TIME = 40  # Moves ghosts are scared
COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY = 1  # Number of points lost each round


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout

    def newGame(self, layout,
                pacmanAgents, ghostAgents,
                display, quiet=False,
                catch_exceptions=False):
        agents = [pacmanAgents] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize(layout, len(ghostAgents))
        game = Game(agents, display,
                    self, catch_exceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.quiet = quiet
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.is_win():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)

    def win(self, state, game):
        if not self.quiet: print("Pacman emerges victorious! Score: %d" % state.data.score)
        game.gameOver = True

    def lose(self, state, game):
        if not self.quiet: print("Pacman died! Score: %d" % state.data.score)
        game.gameOver = True

    def getProgress(self, game):
        return float(game.state.getNumFood()) / self.initialState.getNumFood()

    def agentCrash(self, game, agent_index):
        if agent_index == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    def getMaxTotalTime(self, agent_index):
        return self.timeout

    def getMaxStartupTime(self, agent_index):
        return self.timeout

    def getMoveWarningTime(self, agent_index):
        return self.timeout

    def getMoveTimeout(self, agent_index):
        return self.timeout

    def getMaxTimeWarnings(self, agent_index):
        return 0


class PacmanRules:
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    PACMAN_SPEED = 1

    def get_legal_actions(state):
        """
        Returns a list of possible actions.
        """
        return Actions.getPossibleActions(state.get_pacman_state().
                                          configuration,
                                          state.data.layout.walls)

    get_legal_actions = staticmethod(get_legal_actions)

    def applyAction(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = PacmanRules.get_legal_actions(state)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        pacmanState = state.data.agentStates[0]

        # Update Configuration
        vector = Actions.directionToVector(action, PacmanRules.PACMAN_SPEED)
        pacmanState.configuration = pacmanState.configuration.generate_successor(vector)

        # Eat
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint(next)
        if manhattanDistance(nearest, next) <= 0.5:
            # Remove food
            PacmanRules.consume(nearest, state)

    applyAction = staticmethod(applyAction)

    def consume(position, state):
        x, y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.scoreChange += 10
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position
            numFood = state.getNumFood()
            if numFood == 0 and not state.data._lose:
                state.data.scoreChange += 500
                state.data._win = True
        # Eat capsule
        if position in state.get_capsules():
            state.data.capsules.remove(position)
            state.data._capsuleEaten = position
            # Reset all ghosts' scared timers
            for index in range(1, len(state.data.agentStates)):
                state.data.agentStates[index].scaredTimer = SCARED_TIME

    consume = staticmethod(consume)


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED = 1.0

    def get_legal_actions(state, ghostIndex):
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.getGhostState(ghostIndex).configuration
        possibleActions = Actions.getPossibleActions(conf, state.data.layout.walls)
        reverse = Actions.reverseDirection(conf.direction)
        if Directions.STOP in possibleActions:
            possibleActions.remove(Directions.STOP)
        if reverse in possibleActions and len(possibleActions) > 1:
            possibleActions.remove(reverse)
        return possibleActions

    get_legal_actions = staticmethod(get_legal_actions)

    def applyAction(state, action, ghostIndex):

        legal = GhostRules.get_legal_actions(state, ghostIndex)
        if action not in legal:
            raise Exception("Illegal ghost action " + str(action))

        ghostState = state.data.agentStates[ghostIndex]
        speed = GhostRules.GHOST_SPEED
        if ghostState.scaredTimer > 0: speed /= 2.0
        vector = Actions.directionToVector(action, speed)
        ghostState.configuration = ghostState.configuration.generate_successor(vector)

    applyAction = staticmethod(applyAction)

    def decrement_timer(ghostState):
        timer = ghostState.scaredTimer
        if timer == 1:
            ghostState.configuration.pos = nearestPoint(ghostState.configuration.pos)
        ghostState.scaredTimer = max(0, timer - 1)

    decrement_timer = staticmethod(decrement_timer)

    def checkDeath(state, agent_index):
        pacman_position = state.getPacmanPosition()
        if agent_index == 0:  # Pacman just moved; Anyone can kill him
            for index in range(1, len(state.data.agentStates)):
                ghostState = state.data.agentStates[index]
                ghost_position = ghostState.configuration.getPosition()
                if GhostRules.can_kill(pacman_position, ghost_position):
                    GhostRules.collide(state, ghostState, index)
        else:
            ghostState = state.data.agentStates[agent_index]
            ghost_position = ghostState.configuration.getPosition()
            if GhostRules.can_kill(pacman_position, ghost_position):
                GhostRules.collide(state, ghostState, agent_index)

    checkDeath = staticmethod(checkDeath)

    def collide(state: object, ghostState, agent_index):
        if ghostState.scaredTimer > 0:
            state.data.scoreChange += 200
            GhostRules.placeGhost(state, ghostState)
            ghostState.scaredTimer = 0
            # Added for first-person
            state.data._eaten[agent_index] = True
        else:
            if not state.data._win:
                state.data.scoreChange -= 500
                state.data._lose = True

    collide = staticmethod(collide)

    def can_kill(pacman_position, ghost_position):
        return manhattanDistance(ghost_position, pacman_position) <= COLLISION_TOLERANCE

    can_kill = staticmethod(can_kill)

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start

    placeGhost = staticmethod(placeGhost)


#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
    return str + ' [Default: %default]'


def parseAgentArgs(str):
    if str == None: return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def read_command(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--num_games', dest='num_games', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the map layout'),
                      metavar='LAYOUT_FILE', default='mediumClassic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--textGraphics', action='store_true', dest='textGraphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quietTextGraphics', action='store_true', dest='quietGraphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help=default('the ghost agent TYPE in the ghostAgents module to use'),
                      metavar='TYPE', default='RandomGhost')
    parser.add_option('-k', '--numghosts', type='int', dest='numGhosts',
                      help=default('The maximum number of ghosts to use'), default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom the size of the graphics window'), default=1.0)
    parser.add_option('-f', '--fixRandomSeed', action='store_true', dest='fixRandomSeed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--recordActions', action='store_true', dest='record',
                      help='Writes game histories to a file '
                           '(named by the time they were played)', default=False)
    parser.add_option('--replay', dest='gameToReplay',
                      help='A recorded game file (pickle) to replay', default=None)
    parser.add_option('-a', '--agentArgs', dest='agentArgs',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--num_training', dest='num_training', type='int',
                      help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('--frameTime', dest='frameTime', type='float',
                      help=default('Time to delay between frames; <0 means keyboard'), default=0.1)
    parser.add_option('-c', '--catch_exceptions', action='store_true', dest='catch_exceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help=default
                      ('Maximum length of time an agent can spend '
                       'computing in a single game'), default=30)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood:' + str(otherjunk))
    args = dict()

    # Fix the random seed
    if options.fixRandomSeed: random.seed('cs188')

    # Choose a layout
    args['layout'] = layout.getLayout(options.layout)
    if args['layout'] is None: raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    no_keyboard = options.gameToReplay is None and (options.textGraphics or options.quietGraphics)
    pacman_type = load_agent(options.pacman, no_keyboard)
    agent_opts = parseAgentArgs(options.agentArgs)
    if options.num_training > 0:
        args['num_training'] = options.num_training
        if 'num_training' not in agent_opts: agent_opts['num_training'] = options.num_training
    pacman = pacman_type(**agent_opts)  # Instantiate Pacman with agentArgs
    args['pacman'] = pacman

    # Don't display training games
    if 'numTrain' in agent_opts:
        options.numQuiet = int(agent_opts['numTrain'])
        options.numIgnore = int(agent_opts['numTrain'])

    # Choose a ghost agent
    ghost_type = load_agent(options.ghost, no_keyboard)
    args['ghosts'] = [ghost_type(i + 1) for i in range(options.numGhosts)]

    # Choose a display format
    if options.quietGraphics:
        import text_display
        args['display'] = text_display.NullGraphics()
    elif options.textGraphics:
        import text_display
        text_display.SLEEP_TIME = options.frameTime
        args['display'] = text_display.PacmanGraphics()
    else:
        import graphicsDisplay
        args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime=options.frameTime)
    args['num_games'] = options.num_games
    args['record'] = options.record
    args['catch_exceptions'] = options.catch_exceptions
    args['timeout'] = options.timeout

    # Special case: recorded games don't use the run_games method or args structure
    if options.gameToReplay is not None:
        print('Replaying recorded game %s.' % options.gameToReplay)
        import pickle
        f = open(options.gameToReplay)
        try:
            recorded = pickle.load(f)
        finally:
            f.close()
        recorded['display'] = args['display']
        replay_game(**recorded)
        sys.exit(0)

    return args


def load_agent(pacman, nographics):
    # Looks through all pythonPath Directories for the right module,
    pythonPathStr = os.path.expandvars("$PYTHONPATH")
    if pythonPathStr.find(';') == -1:
        python_path = pythonPathStr.split(':')
    else:
        python_path = pythonPathStr.split(';')
    python_path.append('.')

    for module_dir in python_path:
        if not os.path.isdir(module_dir): continue
        module_names = [f for f in os.listdir(module_dir) if f.endswith('gents.py')]
        for modulename in module_names:
            try:
                module = __import__(modulename[:-3])
            except ImportError:
                continue
            if pacman in dir(module):
                if nographics and modulename == 'keyboardAgents.py':
                    raise Exception('Using the keyboard requires graphics (not text display)')
                return getattr(module, pacman)
    raise Exception('The agent ' + pacman + ' is not specified in any *Agents.py.')


def replay_game(layout, actions, display):
    import pacmanAgents
    import ghostAgents
    rules = ClassicGameRules()
    agents = [pacmanAgents.GreedyAgent()] + [ghostAgents.RandomGhost(i + 1)
                                             for i in
                                             range(layout.getNumGhosts())]
    game = rules.newGame(layout, agents[0], agents[1:], display)
    state = game.state
    display.initialize(state.data)

    for action in actions:
        # Execute the action
        state = state.generate_successor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)

    display.finish()


def file(param):
    pass


def run_games(layout, pacman, ghosts, display, num_games, record, num_training=0,
              catch_exceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = ClassicGameRules(timeout)
    games = []

    for i in range(num_games):
        be_quiet = i < num_training
        if be_quiet:
            # Suppress output and graphics
            import text_display
            game_display = text_display.NullGraphics()
            rules.quiet = True
        else:
            game_display = display
            rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts, game_display, be_quiet, catch_exceptions)
        game.run()
        if not be_quiet:
            games.append(game)

        if record:
            import time
            import pickle
            ('recorded-game-%d' % (i + 1)) + '-'.join([str(t)
                                                       for t in time.localtime()[1:6]])
            file('w')
            components = {'layout': layout, 'actions': game.moveHistory}
            pickle.dump(components, file)
            file.close()

    if (num_games - num_training) > 0:
        scores = [game.state.get_score() for game in games]
        wins = [game.state.is_win() for game in games]
        win_rate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Win Rate:      %d/%d (%.2f)' % (wins.count(True), len(wins), win_rate))
        print('Record:       ', ', '.join([['Loss', 'Win'][int(w)] for w in wins]))

    return games


if __name__ == '__main__':
    args = read_command(sys.argv[1:])
    run_games(**args)
