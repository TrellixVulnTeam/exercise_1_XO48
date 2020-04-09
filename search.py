"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()




def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()




class Node:
    """
    A class which creates nodes which hold the states of problem and which hold information
    about the action to arrive at the node and provide a function to arrive at the nodes children.
    """

    def __init__(self, state, parent, action, step_cost, path_cost=0):
        """
        a class constructor which initializes the node with all the relevant information about
        the following parameters
        :param state: state of the game the node holds
        :param parent: the parent node of the current state, None if there was no parent, Ie the current node is the root
        :param action: the action which brought us to the current node from the parent node
        :param step_cost: the path cost until this node
        """

        self.state = state
        self.parent = parent
        self.action = action
        self.step_cost = step_cost
        self.path_cost = path_cost

    # todo how to compare nodes? Nodes are equal when they share state, parent, action, path cost, step cost? for the
    #  purposes of UCS it may be wise to compare stateand cost and perhaps parent. When we want more complicated
    # comparisons

    def compare(self, compare_to):
        """
        flbsl
        :param node:
        :param compare_to:
        :return:
        """
        if self.state.equals(compare_to.state):
            return lambda x: self.path_cost < compare_to.path_cost   #TODO WTF something like this?





class SearchGraph:
    """
    This class holds the graph of the current problem holding all expanded nodes
    """

    def __init__(self, problem):
        """
        Initialize the graph for the current problem
        :param problem: the problem for the graph
        """

        self.problem = problem

        # build root node
        self.root = Node(problem.get_start_state(), None, None, 0)
        self.vertices = {self.root}
        self.explored_states = {self.root.state}

    def has_node_been_explored(self, node):
        """
        Checks if the nodes state is in the graph
        :param node: the node being checked
        :return: True if the node is in the set
        """
        # TODO for uses until now, this is acceptable, however, as we add path costs we may need to be able to change the parent? create new edges?
        return  node.state in self.explored_states

    def add_node(self, node):
        """
        Add the node to the graph
        :param node: node to be added to the graph
        """
        self.vertices.add(node)
        self.explored_states.add(node.state)

    def get_root(self):
        """
        gets the node which is the initial start state of the problem
        :return: return root node
        """
        return self.root

    def get_actions_to_solution(self, goal_node):
        """
        given a goal state we search backwards to the root to find the list of actions
        that brought us to the given goal.
        :param goal_node: the goal we are trying to arrive at
        :return: a list of actions from the root to the goal node
        """
        actions = []
        current_node = goal_node

        while current_node.parent is not None:
            actions.append(current_node.action)
            current_node = current_node.parent

        actions.reverse()
        return actions



def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    # the graph of the problem
    graph = SearchGraph(problem)

    # checking if the root is a goal state
    root = graph.get_root()
    if problem.is_goal_state(root.state):
        return []

    # create the frontier of nodes to be explored
    fringe = util.Queue()
    fringe.push(root)

    "Iterate through the fringe, as indicated by BFS until we reach a goal state " \
    "or we have not been able to find a goal state "
    while not fringe.isEmpty():
        current_node = fringe.pop()
        if not graph.has_node_been_explored(current_node):
            graph.add_node(current_node)

        for triple in problem.get_successors(current_node.state):
            successor, action, step_cost = triple
            node = Node(successor, current_node, action, step_cost)

            if not graph.has_node_been_explored(node): # todo have to check if it is in the frontier?
                if problem.is_goal_state(node.state):
                    return graph.get_actions_to_solution(node)
                else:
                    fringe.push(node)

    # if a goal state has not been found return an empty list
    return []








def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    graph = SearchGraph(problem)

    root = graph.get_root()
    path_cost = 0

    fringe = util.PriorityQueue() # todo may have to store both states and nodes?
    fringe.push(root, path_cost)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        current_path_cost = current_node.path_cost

        # check if the current node is a goal, otherwise add to the graph
        if problem.is_goal_state(current_node.state):
            return graph.get_actions_to_solution(current_node)

        else:
            graph.add_node(current_node)


            for triple in problem.get_successors(current_node.state):
                successor, action, step_cost = triple
                node = Node(successor, current_node, action, step_cost, current_path_cost + step_cost)

                if not graph.has_node_been_explored(node): # TODO have to check if the state is already in the frontier?
                    fringe.push(node, node.path_cost)

                # todo have to check that it isnt in the frontier with a higher path cost








def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
