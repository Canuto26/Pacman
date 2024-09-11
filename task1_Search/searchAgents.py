# searchAgents.py
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


"""
Este archivo contiene todos los agentes que se pueden seleccionar para controlar Pacman.  A
seleccionar un agente, usar la opción '-p' cuando se ejecuta pacman.py.  Los argumentos pueden ser
se pasa a su agente usando '-a'.  Por ejemplo, para cargar un SearchAgent que usa
depth first search (dfs), ejecute el siguiente comando:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

En el proyecto se pueden encontrar comandos para invocar otras estrategias de búsqueda
descripción.

Por favor, cambie solo las partes del archivo que se le pide.  Busque las líneas
que dicen


"*** YOUR CODE HERE ***"

Las partes que usted llena comienzan aproximadamente 3/4 del camino abajo.  Siga el proyecto
descripción para más detalles.

Buena suerte y feliz búsqueda!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "Un agente que va al oeste hasta que no puede."

    def getAction(self, state):
        "El agente recibe un GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    Este agente de búsqueda muy general encuentra una ruta utilizando una búsqueda proporcionada
    algoritmo para el problema de búsqueda suministrado, a continuación, devuelve acciones para seguir que
    camino.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Las opciones para fn incluyen:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Nota: NO debe cambiar ningún código en SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        Esta es la primera vez que el agente ve el diseño del juego
        board. Aquí, elegimos un camino hacia la meta. En esta fase, el agente
        debe calcular la ruta al objetivo y almacenarla en una variable local.
        Todo el trabajo se hace en este método!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Devuelve la siguiente acción en la ruta elegida anteriormente (en
        registerInitialState).  Devolver Directions.STOP si no hay más
        medidas que deben adoptarse.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    Un problema de búsqueda define el espacio de estado, estado de inicio, prueba de objetivo, sucesor
    función y función de coste.  Este problema de búsqueda puede utilizarse para encontrar rutas
    a un punto particular en la tabla pacman.

    El espacio de estado consiste en (x,y) posiciones en un juego pacman.

    Nota: este problema de búsqueda está completamente especificado; NO debe cambiarlo.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: Un objeto GameState (pacman.py)
        costFn: Una función de un estado de búsqueda (tupla) a un número no negativo
        Objetivo: Una posición en el juego
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Devuelve a los estados sucesores, las acciones que requieren y un costo de 1.

         Como se indica en search.py:
             Para un estado dado, esto debe devolver una lista de triples,
         (sucesor, acción, paso a paso) donde 'sucesor' es
         La acción es el sucesor del estado actual
         El coste de la formación es el coste de la formación continua y
         coste de la ampliación a este sucesor
        """
 
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Devuelve el coste de una secuencia particular de acciones. Si esas acciones
        incluir movimiento ilegal, devolver 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    Un agente de búsqueda de puestos con una función de costes que penaliza la participación en
    posiciones en el lado oeste del tablero.

    La función de coste para entrar en una posición (x,y) es 1/2 x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    Un agente de búsqueda de puestos con una función de costes que penaliza la participación en
    posiciones en el lado este del tablero.

    La función de coste para entrar en una posición (x,y) es 2 x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "La heurística de distancia de Manhattan para una posiciónSearchProblema"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "La heurística de distancia euclidiana para una posiciónSearchProblema"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    Este problema de búsqueda encuentra rutas a través de las cuatro esquinas de un diseño.

    Debe seleccionar un espacio de estado adecuado y una función sucesora
    """

    def __init__(self, startingGameState):
        """
        Almacena las paredes, la posición inicial del pacman y las esquinas.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Devuelve el estado de inicio (en su espacio de estados, no el estado completo de Pacman
        espacio)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Devuelve si este estado de búsqueda es un estado objetivo del problema.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Devuelve a los estados sucesores, las acciones que requieren y un costo de 1.

         Como se indica en search.py:
            Para un estado dado, esto debe devolver una lista de triples, (sucesor,
            acción, stepCost), donde 'sucesor' es un sucesor de la actual
            estado, acción es la acción necesaria para llegar allí y stepCost
            es el costo incremental de la expansión a ese sucesor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Devuelve el costo de una secuencia particular de acciones.  Si esas acciones
        Incluir un traslado ilegal, devolver 999999.  Esto se implementa para usted.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    Una heurística para el problema de cornersProblem que usted definió.

      estado:   El estado actual de la búsqueda
               (una estructura de datos que usted eligió en su problema de búsqueda)

      problema: La instancia CornersProblem para este diseño.

    Esta función siempre debe devolver un número que es un límite inferior en el
    camino más corto desde el estado hasta un objetivo del problema; es decir, debe ser
    admisible (y coherente).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "Un SearchAgent para FoodSearchProblem usando A* y su foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    Un problema de búsqueda asociado con la búsqueda de una ruta que recoge todos los
    comida (puntos) en un juego de Pacman.

    Un estado de búsqueda en este problema es una tupla ( pacmanPosition, foodGrid ) donde
      pacmanPosition: una tupla (x,y) de números enteros que especifica la posición de Pacman
      foodGrid:   una cuadrícula (ver game.py) de Verdadero o Falso, especificando el alimento restante
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Devuelve a los estados sucesores, las acciones que requieren y un costo de 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Devuelve el costo de una secuencia particular de acciones.  Si esas acciones
        incluir movimiento ilegal, devolver 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "Un SearchAgent para FoodSearchProblem usando A* y su foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Tu heurística para el problema de FoodSearchProblem va aquí.

    Esta heurística debe ser coherente para garantizar la corrección.  En primer lugar, trate de llegar
    una heurística admisible; casi todas las heurísticas admisibles serán
    También es coherente.

    Si el uso de A* alguna vez encuentra una solución que es peor que la búsqueda uniforme de costos encuentra,
    su heurística es *no* consistente, y probablemente no admisible!  En el
    por otra parte, la heurística inadmisible o incoherente puede encontrar óptima
    soluciones, así que ten cuidado.

    El estado es una tupla ( pacmanPosition, foodGrid ) donde foodGrid es una cuadrícula
    (ver game.py) de True o False. Puede llamar a foodGrid.asList() para obtener
    una lista de coordenadas alimentarias.

    Si desea acceder a información como paredes, cápsulas, etc., puede consultar la
    problema.  Por ejemplo, problem.walls le da una cuadrícula de dónde están las paredes
    son.

    Si desea *almacenar* la información para reutilizarla en otras llamadas al
    heurística, hay un diccionario llamado problem.heuristicInfo que se puede
    . Por ejemplo, si desea contar las paredes una sola vez y almacenar
    valor, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Las llamadas posteriores a esta heurística pueden acceder
    problema.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    return 0

class ClosestDotSearchAgent(SearchAgent):
    "Buscar todos los alimentos utilizando una secuencia de búsquedas"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Devuelve una ruta (una lista de acciones) al punto más cercano, comenzando desde
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    Un problema de búsqueda para encontrar una ruta a cualquier alimento.

    Este problema de búsqueda es igual que el PositionSearchProblem, pero tiene un
    diferentes pruebas de objetivo, que debe completar a continuación.  El espacio de estado y
    La función sucesora no necesita ser modificada.

    La definición de clase anterior, AnyFoodSearchProblem(PositionSearchProblem),
    hereda los métodos de PositionSearchProblem.

    Puede usar este problema de búsqueda para ayudarle a rellenar el cuadro findPathToClosestDot
    método.
    """

    def __init__(self, gameState):
        "Almacena información del estado de juego.  No es necesario cambiar esto."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        El estado es la posición de Pacman. Rellena esto con una prueba de gol que
        completar la definición del problema.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Devuelve la distancia del laberinto entre dos puntos, utilizando las funciones de búsqueda
    ya ha construido. El gameState puede ser cualquier estado de juego -- Pacman’s
    posición en ese estado se ignora.

    Ejemplo de uso: mazeDistance( (2,4), (5,6), gameState)

    Esta podría ser una función auxiliar útil para su ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
