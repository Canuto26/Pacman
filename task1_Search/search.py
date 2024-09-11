# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    Esta clase describe la estructura de un problema de búsqueda, pero no implementa
    cualquiera de los métodos (en terminología orientada a objetos: una clase abstracta).

    No necesitas cambiar nada en esta clase, nunca.
    """

    def getStartState(self):
        """
        Devuelve el estado inicial del problema de búsqueda.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Devuelve True si y solo si el estado es un estado de objetivo válido.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        Para un estado dado, esto debe devolver una lista de triples, (sucesor,
        acción, stepCost), donde 'sucesor' es un sucesor de la actual
        estado, 'acción' es la acción necesaria para llegar allí, y 'stepCost' es
        el coste incremental de la ampliación a ese sucesor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        Este método devuelve el coste total de una secuencia particular de acciones.
        La secuencia debe estar compuesta por movimientos legales.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Devuelve una secuencia de movimientos que resuelve tinyMaze.  Para cualquier otro laberinto, el
    secuencia de movimientos será incorrecta, por lo que solo se utiliza para tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Buscar los nodos más profundos en el árbol de búsqueda primero.

    Su algoritmo de búsqueda necesita devolver una lista de acciones que llega a la
    objetivo. Asegúrese de implementar un algoritmo de búsqueda de grafos.

    Para empezar, puede que quieras probar algunos de estos comandos simples
    comprender el problema de búsqueda que se pasa en:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = util.Stack()
    trace = util.Stack()

    traveled = []
    step_counter = 0

    start_state = problem.getStartState()
    stack.push((start_state, step_counter, 'START'))

    while not stack.isEmpty():
        
        # llegar al estado
        curr_state, _, action = stack.pop()
        traveled.append(curr_state)
        
        # registrar las acciones que llegan a ese estado
        if action != 'START':
            trace.push(action)
            step_counter += 1

        # comprobar si el estado es la meta
        if problem.isGoalState(curr_state):
            return trace.list

        # obtener posibles estados siguientes
        valid_successors = 0
        successors = problem.getSuccessors(curr_state)

        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # evitar viajar de vuelta a estados anteriores
            if next_state not in traveled:
                valid_successors += 1
                stack.push((next_state, step_counter, next_action))

        # callejón sin salida, paso atrás
        if valid_successors == 0:
            while step_counter != stack.list[-1][1]: # back until next awaiting state
                step_counter -= 1
                trace.pop()
    

def breadthFirstSearch(problem):
    """Buscar los nodos más bajos en el árbol de búsqueda primero."""
    
    queue = util.Queue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    queue.push(start_state)
    seen.append(start_state)

    while not queue.isEmpty():
        
        # llegar al estado
        curr_state = queue.pop()

        # comprobar si el estado es la meta
        if problem.isGoalState(curr_state):
            break

        # obtener posibles estados siguientes
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]

            # avoid traveling back to previous states
            if next_state not in seen:
                seen.append(next_state)
                queue.push(next_state)
                trace[next_state] = (curr_state, next_action)

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions



def uniformCostSearch(problem):
    """Buscar primero el nodo de menor coste total."""
    
    priority_queue = util.PriorityQueue()
    trace = {}
    seen = []

    start_state = problem.getStartState()
    prev_cost = 0
    trace[start_state] = [None, None, prev_cost]

    priority_queue.update(start_state, 0)
    seen.append(start_state)

    while not priority_queue.isEmpty():
        
        # llegar al estado
        curr_state = priority_queue.pop()

        # comprobar si el estado es la meta
        if problem.isGoalState(curr_state):
            break

        # obtener posibles estados siguientes
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]

            # avoid traveling back to previous states
            if next_state not in seen:
                prev_cost = trace[curr_state][2]
                seen.append(next_state)
                priority_queue.update(next_state, next_cost + prev_cost)
                
            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > next_cost + prev_cost:
                    trace[next_state][2] = next_cost + prev_cost
                    trace[next_state][1] = next_action
                    trace[next_state][0] = curr_state
            else:
                trace[next_state] = [curr_state, next_action, next_cost + prev_cost]

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions


def nullHeuristic(state, problem=None):
    """
    Una función heurística estima el coste del estado actual al más cercano
    objetivo en el problema de búsqueda proporcionado.  Esta heurística es trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Buscar el nodo que tiene el menor costo combinado y heurística primero."""
    start_state = problem.getStartState()

    g = {}
    g[start_state] = 0
    def f(curr_node): return float(g[curr_node] + heuristic(curr_node, problem))


    open_list = util.PriorityQueue()
    open_list.push(start_state, 0)
    open_seen = [start_state] # for 'in' operator, as PriorityQueueWithFunction records a tuple with priority
    close_list = []
    trace = {}
    trace[start_state] = [None, None, 0]

    while not open_list.isEmpty():

        # arrive at state
        curr_state = open_list.pop()
        open_seen.remove(curr_state)

        # check if state is goal
        if problem.isGoalState(curr_state):
            break

        # get possible next states
        successors = problem.getSuccessors(curr_state)
        
        for successor in successors:

            next_state = successor[0]
            next_action = successor[1]
            next_cost = successor[2]
            successor_cost = g[curr_state] + next_cost
           
            UPDATE = False
            if next_state in open_seen:
                if g[next_state] <= successor_cost:
                    pass
                else:
                    g[next_state] = successor_cost
                    open_list.update(item=next_state, priority=f(next_state))
            elif next_state in close_list:
                if g[next_state] <= successor_cost:
                    pass
                else: UPDATE = True
            else: UPDATE = True



            if UPDATE:
                g[next_state] = successor_cost
                open_list.update(item=next_state, priority=f(next_state))
                open_seen.append(next_state)

                if next_state in close_list:
                    close_list.remove(next_state)
                    open_seen.remove(next_state)

            # update and allow tracing to the best state
            if next_state in trace:
                if trace[next_state][2] > successor_cost:
                    trace[next_state][0] = curr_state
                    trace[next_state][1] = next_action
                    trace[next_state][2] = successor_cost
            else:
                trace[next_state] = [curr_state, next_action, successor_cost]

        close_list.append(curr_state)

    # back track
    actions = []
    backtrack_state = curr_state # the goal state
    while backtrack_state != start_state:
        prev_state, action, _ = trace[backtrack_state]
        actions.append(action)
        backtrack_state = prev_state
    actions = list(reversed(actions))

    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
