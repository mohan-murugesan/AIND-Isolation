"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Score calculated first based on the number of moves available to the player, 
    with respect to the available moves to the opponent. Board is penalized, 
    if the opponent has more available moves than the player. Also normalizing the difference 
    in moves with the manhattan distance between the player. Board is penalized, if the manhattan 
    distance between the player and the opponent is large, It will be hard for player to 
    block moves for opponent

    This should be the best heuristic function for your project submission.
    The difference in the number of available moves between the current
    player and its opponent one ply ahead in the future is used as the
    score of the current game state
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
    player_legal_moves = game.get_legal_moves(player)
    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))
    
    difference_in_moves = len(player_legal_moves) - 2*len(opp_legal_moves)
    position_of_player = game.get_player_location(player)
    position_of_opponent = game.get_player_location(game.get_opponent(player))
    manhattan_distance = abs(position_of_player[0]-position_of_opponent[0]) +  abs(position_of_player[1]-position_of_opponent[1])

    return(float(difference_in_moves/float(manhattan_distance)))


def custom_score_2(game, player):
    """Computes the difference in legal moves of the two players
    while penalizes player for moving to a corner

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
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Penalize player for moving to corner positions
    corner_weight = 2
    corner_positions = [(0, 0), (0, game.height - 1), (game.width - 1, 0), (game.width - 1, game.height - 1)] 
    if game.get_player_location(player) in corner_positions:
        own_moves -= corner_weight

    return float(own_moves - opp_moves)


def custom_score_3(game, player):
    """Maximize the distance between the player and the opponent. 
    Returns the absolute difference between the sum of
    the location vectors, where larger differences equal higher scores.

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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp_location = game.get_player_location(game.get_opponent(player))
    if opp_location == None:
        return 0.

    own_location = game.get_player_location(player)
    if own_location == None:
        return 0.

    return float(abs(sum(opp_location) - sum(own_location)))


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
        score, move = self.minimax_decision(game, depth)
        return move

    def minimax_decision(self, game, depth, max_player=True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

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
        available_moves = game.get_legal_moves()

        # Recursion Stopping Conditions
        # Terminal/Leaf Node -->No more legal moves available OR
        # Fixed depth search -->Current depth level exceed specified max depth.
        if (not available_moves) or (depth == 0):
            if max_player:
                return (self.score(game, game.active_player), (-1, -1))
            else:
                return (self.score(game, game.inactive_player), (-1, -1))

        if max_player:
            current_score = float("-inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.minimax_decision(game_child, depth-1, max_player=False)

                # Identify the maximum score branch for the current player.
                if child_score >= current_score:
                    current_move = move
                    current_score = child_score

        else:
            current_score = float("inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.minimax_decision(game_child, depth-1, max_player=True)

                # Identify the minimum score branch for the opponent.
                if child_score <= current_score:
                    current_move = move
                    current_score = child_score

        return current_score, current_move


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
            d = 1
            while True:
                best_move = self.alphabeta(game, depth=d)
                # print("Current Depth: ", d, "Best Move: ", best_move)
                d += 1

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        score, move = self.alphabeta_decision(game,depth)
        return move

    def alphabeta_decision(self, game, depth, alpha=float("-inf"), beta=float("inf"), max_player=True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text


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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get all the available moves at the current state
        available_moves = game.get_legal_moves()

        # Recursion Stopping Conditions
        # Terminal/Leaf Node --> No more legal moves available OR
        # Fixed depth search --> Current depth level exceed specified max depth.
        if (not available_moves) or (depth == 0):
            if max_player:
                return (self.score(game, game.active_player), (-1, -1))
            else:
                return (self.score(game, game.inactive_player), (-1, -1))

        current_move = available_moves[0]
        if max_player:
            current_score = float("-inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.alphabeta_decision(game_child, depth-1,
                                                           alpha, beta,
                                                           False)

                if child_score >= current_score:
                    current_score = child_score

                # Test if the branch utility is greater than beta,
                # then prune other sibling branches.
                if current_score >= beta:
                    return current_score, move

                # Update alpha if branch utility is greater than
                # current value of alpha for MAX nodes.
                if current_score > alpha:
                    current_move = move
                    alpha = current_score

        else:
            current_score = float("inf")
            for move in available_moves:
                game_child = game.forecast_move(move)
                child_score, child_move = self.alphabeta_decision(game_child, depth-1,
                                                           alpha, beta,
                                                           True)

                if child_score <= current_score:
                    current_score = child_score

                # Test if the branch utility is less than alpha,
                # then prune other sibling branches.
                if current_score <= alpha:
                    return current_score, move

                # Update beta if branch utility is lesser than
                # current value of beta for MIN nodes.
                if current_score < beta:
                    current_move = move
                    beta = current_score

        return current_score, current_move
