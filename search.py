"""
In search.py, you will implement generic search algorithms
"""

import util
import numpy


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
        :param parent: the parent of the current, null if there was no parent
        :param action: the action which brought us to the current node from the parent node
        :param step_cost: the cost from the previous node to this node
        :param path_cost: the path cost until this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.step_cost = step_cost
        self.path_cost = path_cost

    def find_actions(self):
        actions = list()
        while self.parent is not None:
            actions.append(self.action)
            self = self.parent

        actions.reverse()

        return actions

    def get_path_cost(self):
        """ returns the nodes path cost"""
        return self.path_cost


def depth_first_search(problem):
    root = problem.get_start_state()
    if problem.is_goal_state(root):
        return []

    root_node = Node(root, None, 0, 0)
    fringe = util.Stack()
    fringe.push(root_node)
    visited = {root_node.state}

    while not fringe.isEmpty():
        curr_node = fringe.pop()

        if problem.is_goal_state(curr_node.state):
            return curr_node.find_actions()

        for triple in problem.get_successors(curr_node.state):
            successor, action, step_cost = triple
            child = Node(successor, curr_node, action, step_cost)

            if child.state not in visited:
                visited.add(child.state)
                fringe.push(child)

    return []


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    root = problem.get_start_state()
    if problem.is_goal_state(root):
        return []

    root_node = Node(root, None, None, 0)
    fringe = util.Queue()
    visited = {root_node}
    fringe.push(root_node)

    while not fringe.isEmpty():
        curr_node = fringe.pop()
        if curr_node.state not in visited:
            visited.add(curr_node.state)

        for triple in problem.get_successors(curr_node.state):
            successor, action, step_cost = triple
            node = Node(successor, curr_node, action, step_cost)

            if node.state not in visited:
                if problem.is_goal_state(node.state):
                    return node.find_actions()
                else:
                    fringe.push(node)

    return []


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    return a_star_search(problem)


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

    fringe = util.PriorityQueueWithFunction(
        lambda y: y.get_path_cost() + heuristic(y.state, problem))

    root_node = Node(problem.get_start_state(), None, None, 0, 0)

    fringe.push(root_node)
    visited = set()

    while not fringe.isEmpty():
        current_node = fringe.pop()
        path_cost = current_node.get_path_cost()

        if problem.is_goal_state(current_node.state):
            return current_node.find_actions()

        if current_node.state not in visited:
            visited.add(current_node.state)

            for triple in problem.get_successors(current_node.state):
                successor, action, step_cost = triple
                fringe.push(Node(successor, current_node, action, step_cost, path_cost + step_cost))

    return []



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
