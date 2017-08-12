"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from random import randint
import math



class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """
    Strategy
    --------
    Uses a normalized feautre list consisting of features around relative number
    of open moves and relative centerness for the players, weighting the number
    of open moves heavier. Additionally, it only factors in centerness during the
    first half of the game.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = number_moves(game, player) / 8
    if own_moves == 0:
        return float("-inf")

    opp_moves = number_moves(game, game.get_opponent(player)) / 8
    if opp_moves == 0:
        return float("inf")

    move_ratio = (own_moves * 8) / (opp_moves * 8) / 8

    # Calculate centerness_score
    completeness = completeness_of_game(game)
    centerness_score = 0
    if completeness < 0.5:
        centerness_max = (game.width / 2.)**2 + (game.height / 2.)**2

        own_centerness = centerness(game, player) / centerness_max
        opp_centerness = centerness(game, game.get_opponent(player)) / centerness_max
        centerness_ratio = (own_centerness * centerness_max) / (centerness_max * opp_centerness + 0.1) / centerness_max

        centerness_score = -1 * own_centerness + opp_centerness - centerness_ratio

    return 2 * own_moves - 2 * opp_moves + 2 * move_ratio + centerness_score


def custom_score_2(game, player):
    """
    Strategy
    --------
    Similar to custom_score, but doesn't normalize the feature values before
    applying weights.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = number_moves(game, player)
    if own_moves == 0:
        return float("-inf")

    opp_moves = number_moves(game, game.get_opponent(player))
    if opp_moves == 0:
        return float("inf")

    move_ratio = own_moves / opp_moves

    completeness = completeness_of_game(game)
    centerness_score = 0

    if completeness < 0.5:
        own_centerness = centerness(game, player)
        opp_centerness = centerness(game, game.get_opponent(player))
        centerness_ratio = own_centerness / opp_centerness + 0.1

        center_score = -1 * own_centerness + opp_centerness - centerness_ratio

    return 2 * own_moves - 2 * opp_moves + 2 * move_ratio + centerness_score


def custom_score_3(game, player):
    """
    Strategy
    --------
    Uses the simple non-normalized ratio between the player's open moves as the
    evaluation function. No additional features or weights applied.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = number_moves(game, player)
    if own_moves == 0:
        return float("-inf")

    opp_moves = number_moves(game, game.get_opponent(player))
    if opp_moves == 0:
        return float("inf")

    #Between 1-8
    return own_moves / opp_moves



def custom_score_general(game, player, constants=[]):
    """
    A general scoring function that accepts a list of constants to be applied
    to the various features as relative weights.

    List of features:
     * own_moves
     * opp_moves
     * move_ratio
     * own_openness
     * opp_openness
     * openness_ratio
     * own_centerness
     * opp_centerness
     * centerness_ratio

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    constants : list(numeric)
        A list of numeric constants to be applied to the features as relative
        weights
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    v = []

    if constants[0] != 0 or constants[2] != 0:
        own_moves = number_moves(game, player) / 8

        if own_moves == 0:
            return float("-inf")

        v.append(own_moves)

    if constants[1] != 0 or constants[2] != 0:
        opp_moves = number_moves(game, game.get_opponent(player)) / 8

        if opp_moves == 0:
            return float("inf")

        v.append(opp_moves)

    if constants[2] != 0:
        move_ratio = (own_moves * 8) / (opp_moves * 8) / 8
        v.append(move_ratio)

    if constants[3] != 0 or constants[5] != 0:
        own_openness = nearby_openness(game, player) / 80
        v.append(own_openness)

    if constants[4] != 0 or constants[5] != 0:
        opp_openness = nearby_openness(game, game.get_opponent(player)) / 80
        v.append(opp_openness)

    if constants[5] != 0:
        openness_ratio = (own_openness * 80) / (opp_openness + 0.0001 * 80) /80
        v.append(openness_ratio)

    centerness_max = (game.width / 2.)**2 + (game.height / 2.)**2

    if constants[6] != 0 or constants[8] != 0:
        own_centerness = centerness(game, player) / centerness_max
        v.append(own_centerness)

    if constants[7] != 0 or constants[8] != 0:
        opp_centerness = centerness(game, game.get_opponent(player)) / centerness_max
        v.append(opp_centerness)

    if constants[8] != 0:
        centerness_ratio = (own_centerness * centerness_max) / (centerness_max * opp_centerness + 0.1) / centerness_max

    return sum([x * y for x, y in zip(constants, v)])


def custom_score_general2(game, player, constants=[]):
    """
    A general scoring function that accepts a list of constants to be applied
    to the various features as relative weights.

    List of features:
     * own_moves
     * opp_moves
     * move_ratio
     * apply centerness based on completeness
     * own_centerness
     * opp_centerness
     * centerness_ratio

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    constants : list(numeric)
        A list of numeric constants to be applied to the features as relative
        weights
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = number_moves(game, player) / 8
    if own_moves == 0:
        return float("-inf")

    opp_moves = number_moves(game, game.get_opponent(player)) / 8
    if opp_moves == 0:
        return float("inf")

    move_ratio = (own_moves * 8) / (opp_moves * 8) / 8

    # Calculate centerness_score
    completeness = completeness_of_game(game)
    centerness_score = 0
    if completeness < constants[3]:
        centerness_max = (game.width / 2.)**2 + (game.height / 2.)**2

        own_centerness = centerness(game, player) / centerness_max
        opp_centerness = centerness(game, game.get_opponent(player)) / centerness_max
        centerness_ratio = (own_centerness * centerness_max) / (centerness_max * opp_centerness + 0.1) / centerness_max

        centerness_score = constants[4] * own_centerness + constants[5] * opp_centerness + constants[6] * centerness_ratio

    return constants[0] * own_moves + constants[1] * opp_moves + constants[2] * move_ratio + centerness_score



def number_moves(game, player):
    """Calculate the number of available moves for the passed in player

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    -------
    float
        The number of available moves for the passed in player.
    """
    return float(len(game.get_legal_moves(player)))


def nearby_openness(game, player):
    """
    Returns
    -------
    float
        Number of avalabls squares in a 4 square radius from the player's
        current position. Result will be between 0-80.
    """
    radius = 4
    nearby_legal_move_count = 0

    current_location = game.get_player_location(player)

    min_row = current_location[0] - radius
    max_row = current_location[0] + radius + 1
    min_col = current_location[1] - radius
    max_col = current_location[1] + radius + 1

    for row in range(min_row, max_row):
        if row < 0 or row > game.height - 1:
            continue

        for col in range(min_col, max_col):
            if col < 0 or col > game.width - 1:
                continue

            if game.move_is_legal((row, col)):
                nearby_legal_move_count += 1

    # print()
    # print(player)
    # print(nearby_legal_move_count)
    # print(game.to_string())

    return float(nearby_legal_move_count)


def centerness(game, player):
    """Outputs a score equal to square of the distance from the center of the
    board to the position of the player.

    This heuristic is only used by the autograder for testing.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


def completeness_of_game(game):
    """A measuer of how complete the board is.

    Returns
    -------
    float
        The percent of complete the game board is. Between 0 and 1.
    """
    spaces = game.width * game.height
    played_spaces = len([x for x in game._board_state[:-3] if x == 1])
    return float(played_spaces / spaces)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.timeout_depths = []

    def average_timeout_depth(self):
        """Returns the average timeout depth-limited
        """
        if self.timeout_depths:
            return sum(self.timeout_depths) / len(self.timeout_depths)
        else:
            return -1


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        """
        From AIMA psuedocode:

        function MINIMAX-DECISION(state) returns an action
            return arg max a is in ACTIONS(s) MIN-VALUE(RESULT(state, a))
        """

        best_move = (-1,-1)
        best_score = float("-inf")
        actions = game.get_legal_moves()

        if not actions:
            return best_move
        else:
            best_move = actions[randint(0, len(actions) - 1)]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # return max(actions, key=lambda action: self._min_value(game.forecast_move(action), 1))
            for action in actions:
                score = self._min_value(game.forecast_move(action), 1)
                if score > best_score:
                    best_score = score
                    best_move = action

        except SearchTimeout:
            pass

        return best_move


    def _min_value(self, game, depth):
        """
        From AIMA psuedocode:

        function MIN-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then return UTILITY(state)
            v = infinity
            for each a in ACTIONS(state) do
                v = MIN(v, MAX-VALUE(RESULT(state, a)))
            return v
        """
        if self._terminal_test(game, depth):
            return self.score(game, self)
        else:
            v = float("inf")

            for action in game.get_legal_moves():
                v = min(v, self._max_value(game.forecast_move(action), depth + 1))

            return v


    def _max_value(self, game, depth):
        """
        From AIMA psuedocode:

        function MAX-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then return UTILITY(state)
            v = -infinity
            for each a in ACTIONS(state) do
                v = MAX(v, MIN-VALUE(RESULT(state, a)))
            return v
        """
        if self._terminal_test(game, depth):
            return self.score(game, self)
        else:
            v = float("-inf")

            for action in game.get_legal_moves():
                v = max(v, self._min_value(game.forecast_move(action), depth + 1))

            return v


    def _terminal_test(self, game, depth):
        """
        Check if the depth is equal or greater than the search_depth of the
        agent or if there are no legal moves.

        Raise SearchTimeout if time_left is less than the TIMER_THRESHOLD.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            self.timeout_depths.append(depth)
            raise SearchTimeout()

        beyond_search_depth = depth >= self.search_depth
        no_legal_moves = len(game.get_legal_moves()) == 0

        return beyond_search_depth or no_legal_moves



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            self.search_depth = 0

            while self.time_left() > self.TIMER_THRESHOLD:
                self.search_depth += 1
                best_move = self.alphabeta(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # print("AlphaBetaPlayer.alphabeta")
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        """
        From AIMA psuedocode:

        function ALPHA-BETA-SEARCH(state) returns an action
            v = MAX-VALUE(state, -infinity, infinity)

        return the action in ACTIONS(state) with value v
        """

        best_move = (-1, -1)
        actions = game.get_legal_moves()

        if not actions:
            return best_move
        else:
            best_move = actions[randint(0, len(actions) - 1)]

        # sorting moves, to facilitate better testing. Will mock it if it becomes
        # problematic
        # try:
        #     # The try/except block will automatically catch the exception
        #     # raised when the timer is about to expire.
        for action in sorted(actions):
            v = self._min_value(game.forecast_move(action), alpha, beta, 1)

            # print("v = {}".format(v))
            if v > alpha:
                alpha = v
                best_move = action

        # except SearchTimeout:
        #     # print("SearchTimeout in AlphaBetaPlayer.alphabeta. best_move = {}".format(best_move))
        #     pass  # Handle any actions required after timeout as needed

        # print("returning best_move: {}".format(best_move))
        return best_move

    def _max_value(self, game, alpha, beta, depth):
        """
        From AIMA psuedocode:

        function MAX-VALUE(state, alpha, beta) returns a utility value
            if TERMINAL-TEST(state) the return UTILITY(state)

            v = -infinity

            for each a in ACTIONS(state) do
            v = MAX(v, MIN-VALUE(RESULT(state, a), alpha, beta))
                if v >= beta then return v
                alpha = MAX(alpha, v)
            return v
        """
        if self._terminal_test(game, depth):
            return self.score(game, self)
        else:
            v = float("-inf")

            for action in game.get_legal_moves():
                min_value = self._min_value(game.forecast_move(action), alpha, beta, depth + 1)

                v = max(v, min_value)

                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v


    def _min_value(self, game, alpha, beta, depth):
        """
        From AIMA psuedocode:

        function MIN-VALUE(state, alpha, beta) returns a utility value
            if TERMINAL-TEST(state) the return UTILITY(state)

            v = infinity

            for each a in ACTIONS(state) do
                v = MIN(v, MAX-VALUE(RESULT(state, a), alpha, beta))
                if v <= alpha then return v
                beta = MIN(beta, v)

            return v
        """
        if self._terminal_test(game, depth):
            return self.score(game, self)
        else:
            v = float("inf")

            for action in game.get_legal_moves():
                max_value = self._max_value(game.forecast_move(action), alpha, beta, depth + 1)

                v = min(v, max_value)

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v

    def _terminal_test(self, game, depth):
        """
        Check if the depth is equal or greater than the search_depth of the
        agent or if there are no legal moves.

        Raise SearchTimeout if time_left is less than the TIMER_THRESHOLD.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            self.timeout_depths.append(depth)
            raise SearchTimeout()

        beyond_search_depth = depth >= self.search_depth
        no_legal_moves = len(game.get_legal_moves()) == 0

        return beyond_search_depth or no_legal_moves
