from board import Board, Move
from search import SearchProblem, ucs
import util
import numpy as np


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
        return [(state.do_move(0, move), move, 1) for move in
                state.get_legal_moves(0)]

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
        corner_values = [state.get_position(0, 0), state.get_position(0, -1),
                         state.get_position(-1, 0),
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
        required to get there, stepCostand '' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

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
    corners = [(0, 0), (state.board_w - 1, 0,), (0, state.board_h - 1),
               (state.board_w - 1, state.board_h - 1)]

    uncovered_corners = find_uncovered_targets(state, corners)

    if len(uncovered_corners) == 0:
        return 0

    unused_pieces = find_unused_pieces(state)

    if len(unused_pieces) == 0:
        return 10934875010193458109345

    min_tiles = find_size_of_smallest_piece(unused_pieces)

    corners_left = len(uncovered_corners)
    min_dist = min_distance_from_target_heuristic(state, uncovered_corners)
    winnable = is_winnable_position(corners, state)
    return max(0, corners_left, min_dist, min_tiles, winnable)

def find_size_of_smallest_piece(pieces):
    return min(piece.get_num_tiles() for piece in pieces )

def min_distance_from_target_heuristic(state, uncovered_targets):
    # a list of indexes where tiles have been placed.
    non_empty_tiles = np.argwhere(state.state > -1)

    return min(min_distance_to_target(coordinate, non_empty_tiles)
               for coordinate in uncovered_targets)

def min_distance_to_target(target, non_zero_indices):

    if len(non_zero_indices) != 0:
        return min(calculate_distance(coordinate, target)
               for coordinate in non_zero_indices)

    return 1

def winnable_position(target, state):
    """
    try to rank whether we can win from a certain position. We check to see if the blocks surrounding a target point
    have been filled in such a way as to make it impossible to win from the current position. Ie if the edges of the
    target point have been covered.
    :param targets: the targets which remain to be covered
    :return: 0 if it is possible to win from the current position otherwise one
    """
    # how to check all the surrounding blocks.

    if target[0] + 1 < state.board_w:
        if state.get_position(target[1], target[0] + 1) != -1:  # the tile above
            return False
    if target[0] - 1 >= 0:
        if state.get_position(target[1], target[0] - 1) != -1: # the tile below
            return False
    if target[1] + 1 < state.board_h: # the tile to the right
        if state.get_position(target[1] + 1, target[0]) != -1:
            return False
    if target[1] - 1 >= 0: # the tile to the left
        if state.get_position(target[1] - 1, target[0]) != -1:
            return False

    return True

def is_winnable_position(targets, state):
    bad_state_const = 10934875010193458109345
    for target in targets:
        if not winnable_position(target, state):
            return bad_state_const

    return 0


def calculate_distance(coordinate, target):
    return max(abs(target[0] - coordinate[0]), abs(target[1] - coordinate[1])) - 1


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        for target in self.targets:
            state_x = state.get_position(target[1], target[0])

            if state_x == -1:
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
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

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


def blokus_cover_heuristic(state, problem):
    uncovered_targets = find_uncovered_targets(state, problem.targets)
    if len(uncovered_targets) == 0:
        return 0

    unused_pieces = find_unused_pieces(state)

    if len(unused_pieces) == 0:
        return 10934875010193458109345

    min_tiles = find_size_of_smallest_piece(unused_pieces)
    min_dist = min_distance_from_target_heuristic(state, uncovered_targets)
    winnable = is_winnable_position(uncovered_targets, state)
    return max(0, len(uncovered_targets), min_dist, min_tiles, winnable)


def num_of_uncovered_targets(state, problem):
    num_points = 0
    for target in problem.targets:
        if state.get_position(target[1], target[0]) == -1:
            num_points += 1

    return num_points



class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"
        self.starting_point = starting_point
        self.board_w = board_w
        self.board_h = board_h
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state, target):

        if state.state[target[0], target[1]] == -1:
            return False

        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, stepCostand '' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

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

        backtrace = []
        current_state = self.board.__copy__()

        self.targets.sort(key=lambda x: calculate_distance(self.starting_point, x))

        for target in self.targets:

            actions = self.greedy_best_first_search(current_state, target)

            for action in actions:
                current_state = current_state.do_move(0, action)
                backtrace.append(action)

        return backtrace

    def greedy_best_first_search(self, current_state, target):

        # create the root
        root = current_state
        root_node = Node(root, [])

        # set up the frontier and visited
        frontier = util.PriorityQueueWithFunction(lambda x: blokus_cover_heuristic(x.state, self))
        visited = set()
        frontier.push(root_node)

        while not frontier.isEmpty():

            current_node = frontier.pop()

            if self.is_goal_state(current_node.state, target):
                return current_node.get_actions()

            if current_node.state not in visited:
                visited.add(current_node.state)

                for triple in self.get_successors(current_node.state):
                    successor, action, step_cost = triple
                    node = Node(successor, current_node.actions + [action])
                    frontier.push(node)

        return []

class Node:

    def __init__(self, state, actions):
        self.state = state  # the nodes state
        #  self.parent = parent  # the nodes parent node or state?
          # a string representation of the boards state
        self.actions = actions


    def get_actions(self):
        return self.actions


def find_unused_pieces(state):
    unused_pieces = set()
    for piece in state.piece_list:
        if state.pieces[0, state.piece_list.pieces.index(piece)]:  # unused piece
            unused_pieces.add(piece)
    return unused_pieces


def find_uncovered_targets(state, targets):
    uncovered_targets = set()
    for target in targets:
        if state.get_position(target[1], target[0]) == -1:  # uncovered
            uncovered_targets.add(target)
    return uncovered_targets


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.targets = targets.copy()
        self.expanded = 0
        self.starting_point = starting_point
        self.board_w = board_w
        self.board_h = board_h
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, stepCostand '' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def is_goal_state(self, state):

        for target in self.targets:
            if state.state[target[0], target[1]] == -1:
                return False

        return True

    def solve(self):
        "*** YOUR CODE HERE ***"
        # run greedy best first search
        return self.greedy_best_first_search()

    def greedy_best_first_search(self):

        root = self.get_start_state()
        root_node = Node(root, [])

        # set up the frontier and visited
        frontier = util.PriorityQueueWithFunction(lambda x: blokus_cover_heuristic(x.state, self))
        visited = set()
        frontier.push(root_node)

        while not frontier.isEmpty():

            current_node = frontier.pop()

            if self.is_goal_state(current_node.state):
                return current_node.get_actions()

            if current_node.state not in visited:
                visited.add(current_node.state)

                for triple in self.get_successors(current_node.state):
                    successor, action, step_cost = triple
                    frontier.push(Node(successor, current_node.actions + [action]))

        return []