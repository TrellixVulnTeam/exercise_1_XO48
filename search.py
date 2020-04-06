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

    def __init__(self, state, parent, action, path_cost):
        """
        a class constructor which initializes the node with all the relevant information about
        the following parameters
        :param state: state of the game the node holds
        :param parent: the parent of the current, null if there was no parent
        :param action: the action which brought us to the current node from the parent node
        :param path_cost: the path cost until this node
        """

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost


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
        self.root = Node(problem.get_start_state().state, None, None, 0)
        self.vertices = {self.root}

    def has_node_been_explored(self, node):
        """
        Checks if the nodes state is in the graph
        :param node: the node being checked
        :return: True if the node is in the set
        """
        return node in self.vertices

    def add_node(self, node):
        """
        Add the node to the graph
        :param node: node to be added to the graph
        """
        self.vertices.add(node)

    def get_root(self):
        """
        gets the node which is the initial start state of the problem
        :return: return root node
        """
        return self.root

    def get_path_to_current_node(self, goal_node):
        """"""
        actions = list()
        current_node = goal_node
        while current_node != self.root:
            path += current_node.parent
            current_node = current_node.parent

        return path.reverse()









def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # the graph of the problem
    graph = SearchGraph(problem)

    # checking if the root is a goal state
    root = graph.get_root()
    if problem.is_goal_state(root.state):
        return []

    # create the frontier
    fringe = util.Queue()
    fringe.push(root.state)

    while not fringe.isEmpty():
        current_node = fringe.pop()
        if not graph.has_node_been_explored(current_node):
            graph.add_node(current_node)

        for triple in problem.get_successors():
            state, action, step_cost = triple
            node = Node(state, current_node, action, step_cost)

            if (not graph.has_node_been_explored(node)) and (node.state not in fringe):
                if problem.is_goal_state(node.state):
                    return #TODO what do we return? Path to the current node?

                fringe.push(node)










def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


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
