import unittest
import unittest.mock as mock

import isolation
import game_agent

from importlib import reload


class MinimaxTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer(search_depth=3)
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def test_depth_is_limited(self):
        with mock.patch.object(self.player1, '_terminal_test', wraps=self.player1._terminal_test) as terminal_test:
            self.player1.get_move(self.game, lambda : 15.)
            terminal_tests_with_depth_4 = [c for c in terminal_test.call_args_list if c[0][1] == 4]
            self.assertEqual(len(terminal_tests_with_depth_4), 0)

    def test_minimax_raises_searchtimeout(self):
        self.player1.time_left = lambda : 0.
        with self.assertRaises(game_agent.SearchTimeout):
            self.player1.minimax(self.game, 3)

    def test_get_move_returns_illegal_move_if_timeout(self):
        best_move = self.player1.get_move(self.game, lambda : 0.)
        self.assertEqual(best_move, (-1,-1))

    def test_minimax_scores_expected_number_of_nodes(self):
        self.player1 = game_agent.MinimaxPlayer(search_depth=2)
        self.game = isolation.Board(self.player1, self.player2, width=9, height=9)
        self.game._board_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 32]

        with mock.patch.object(self.player1, 'score', wraps=self.player1.score) as score_fn:
            self.player1.get_move(self.game, lambda : 15.)
            self.assertEqual(score_fn.call_count, 35)


if __name__ == '__main__':
    unittest.main()
