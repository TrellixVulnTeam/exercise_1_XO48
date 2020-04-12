from board import Board
from search import SearchProblem, ucs
import util
import numpy as np
import math

class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        # todo if the starting position is not (0,0) rotate the board until it is?

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        checks whether all the corners have been filled in. By accessing the numpy array and its corner values then
        checking to see whether any of those corner values have been filled
        :param state: a numpy array representing the current state of the board
        :return: False if any of the corners are still empty, otherwise True
        """
        corner_values = [state.get_position(0, 0), state.get_position(0, -1), state.get_position(-1, 0),
                         state.get_position(-1, -1)]

        if -1 in corner_values:
            return False

        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """

        num_of_tiles = 0
        for move in actions:
            num_of_tiles += move.piece.get_num_tiles()

        return num_of_tiles



def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    return max(how_many_corners_heuristic(state, problem), 0, min_distance_from_corners_heuristic(state))


def how_many_corners_heuristic(state):
    """
    This is a heuristic which simply checks how many corners have not yet been filled and assumes the number of
    unfilled corners is the quickest solution. A very optimistic heuristic.
    :param state: the current state of the problem
    :return: how many moves the heuristic estimates until the corners are filled
    """
    corner_values = [state.get_position(0, 0), state.get_position(0, -1), state.get_position(-1, 0),
                     state.get_position(-1, -1)]

    return corner_values.count(-1)


def min_distance_from_corners_heuristic(state):
    """
    calculates the min number of tiles that would be used from each corner to the nearest signed tile
    :param state: the state of the board
    :return: the min number of tiles that would have to be used
    """

    # a list of indexes where tiles have been placed.
    non_empty_tiles = np.argwhere(state >= 0)

    # min number of tiles
    min_tiles = 0

    if state.get_position(0, 0) == -1:  # upper left corner
        min_tiles += minimum_num_tiles(0, 0, non_empty_tiles)
    if state.get_position(0, state.board_w - 1) == -1:  # upper right corner
        min_tiles += minimum_num_tiles(0, state.board_w - 1, non_empty_tiles)
    if state.get_position(state.board_h - 1, 0) == -1:  # lower left corner
        min_tiles += minimum_num_tiles(state.board_h - 1, 0, non_empty_tiles)
    if state.get_position(state.board_h - 1, state.board_w - 1) == -1:  # lower right corner
        min_tiles += minimum_num_tiles(state.board_h - 1, state.board_w - 1, non_empty_tiles)

    return min_tiles


def minimum_num_tiles(x_corner, y_corner, non_zero_indices):
    """
    looks for the minimum number of tiles that would have to be signed between the given corner and all of the signed
    tiles and returns the minumum required
    :param x_corner: the x coordinate
    :param y_corner: the y coordinate
    :param non_zero_indices: an array of indices which have been used thus far
    :return: the minimum number of tiles that would need to be used between the given corner and the closest signed tile
    """

    minimum = 0
    for coordinates in non_zero_indices:
        n = math.abs(x_corner - coordinates[0])
        m = math.abs(y_corner - coordinates[1])

        num_diagonal_tiles = n + m - math.gcd(n, m)
        if minimum >= num_diagonal_tiles:
            minimum = num_diagonal_tiles

    return minimum



class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
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
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()



class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

