"""Estimate the strength rating of a student defined heuristic by competing
against fixed-depth minimax and alpha-beta search agents in a round-robin
tournament.

NOTE: All agents are constructed from the student CustomPlayer implementation,
so any errors present in that class will affect the outcome.

The student agent plays a number of "fair" matches against each test agent.
The matches are fair because the board is initialized randomly for both
players, and the players play each match twice -- once as the first player and
once as the second player.  Randomizing the openings and switching the player
order corrects for imbalances due to both starting position and initiative.
"""
import itertools
import random
import warnings

from collections import namedtuple
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

from isolation import Board
from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)
from game_agent import (MinimaxPlayer, AlphaBetaPlayer, custom_score,
                        custom_score_2, custom_score_3,
                        custom_score_general, custom_score_general2)

NUM_PROCS = 4
NUM_MATCHES = 100  # number of matches against each opponent
TIME_LIMIT = 150  # number of milliseconds before timeout

DESCRIPTION = """
This script evaluates the performance of the custom_score evaluation
function against a baseline agent using alpha-beta search and iterative
deepening (ID) called `Improved`. The three `Custom` agents use
ID and alpha-beta search with the custom_score functions defined in
game_agent.py.
"""

Agent = namedtuple("Agent", ["player", "name"])

def _run(*args):
    idx, p1, p2, moves = args[0]
    game = Board(p1, p2)
    for m in moves:
        game.apply_move(m)
    winner, _, termination = game.play(time_limit=TIME_LIMIT)

    try:
        p1_avg_timeout_depth = game._player_1.average_timeout_depth()
    except AttributeError:
        p1_avg_timeout_depth = -1

    try:
        p2_avg_timeout_depth = game._player_2.average_timeout_depth()
    except AttributeError:
        p2_avg_timeout_depth = -1


    return (idx, winner == p1), termination, p1_avg_timeout_depth, p2_avg_timeout_depth


def play_round(cpu_agent, test_agents, win_counts, num_matches, average_timeout_depths):
    """Compare the test agents to the cpu agent in "fair" matches.

    "Fair" matches use random starting locations and force the agents to
    play as both first and second player to control for advantages resulting
    from choosing better opening moves or having first initiative to move.
    """
    timeout_count = 0
    forfeit_count = 0
    pool = Pool(NUM_PROCS)

    for _ in range(num_matches):

        # initialize all games with a random move and response
        init_moves = []
        init_game = Board("p1", "p2")
        for _ in range(2):
            move = random.choice(init_game.get_legal_moves())
            init_moves.append(move)
            init_game.apply_move(move)

        games = sum([[(2 * i, cpu_agent.player, agent.player, init_moves),
                      (2 * i + 1, agent.player, cpu_agent.player, init_moves)]
                    for i, agent in enumerate(test_agents)], [])

        # play all games and tally the results
        for result, termination, p1_avg_timeout_depth, p2_avg_timeout_depth in pool.imap_unordered(_run, games):
            game = games[result[0]]
            winner = game[1] if result[1] else game[2]

            win_counts[winner] += 1

            if termination == "timeout":
                print("TIMEOUT: {}".format(game))
                timeout_count += 1
            elif winner not in test_agents and termination == "forfeit":
                print("FORFEIT: {}".format(game))
                forfeit_count += 1

            average_timeout_depths[game[1]] = p1_avg_timeout_depth
            average_timeout_depths[game[2]] = p2_avg_timeout_depth

    return timeout_count, forfeit_count


def update(total_wins, wins):
    for player in total_wins:
        total_wins[player] += wins[player]
    return total_wins


def play_matches(cpu_agents, test_agents, num_matches):
    """Play matches between the test agent and each cpu_agent individually. """
    total_wins = {agent.player: 0 for agent in test_agents}
    average_timeout_depths = {agent.player: -1 for agent in test_agents}
    total_timeouts = 0.
    total_forfeits = 0.
    total_matches = 2 * num_matches * len(cpu_agents)

    # print("\n{:^9}{:^13}{:^13}{:^13}{:^13}{:^13}".format(
    #     "Match #", "Opponent", test_agents[0].name, test_agents[1].name,
    #     test_agents[2].name, test_agents[3].name))
    print("\n{:^9}{:^13}".format("Match #", "Opponent") + ''.join(['{:^15}'.format(x[1].name) for x in enumerate(test_agents)]))

    # print("{:^9}{:^13} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5}"
    #       .format("", "", *(["Won", "Lost"] * 4)))
    print("{:^9}{:^13} ".format("", "") +  ' '.join(['{:^6}| {:^6}'.format("Won", "Lost") for x in enumerate(test_agents)]))


    for idx, agent in enumerate(cpu_agents):
        # wins = {test_agents[0].player: 0,
        #         test_agents[1].player: 0,
        #         test_agents[2].player: 0,
        #         test_agents[3].player: 0,
        #         agent.player: 0}
        wins = {key: 0 for (key, value) in test_agents}
        wins[agent.player] = 0

        print("{!s:^9}{:^13}".format(idx + 1, agent.name), end="", flush=True)

        counts = play_round(agent, test_agents, wins, num_matches, average_timeout_depths)
        # print(average_timeout_depths)
        total_timeouts += counts[0]
        total_forfeits += counts[1]
        total_wins = update(total_wins, wins)
        _total = 2 * num_matches
        round_totals = sum([[wins[agent.player], _total - wins[agent.player]]
                            for agent in test_agents], [])
        # print(" {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5}"
        #       .format(*round_totals))
        print(' ' + ' '.join([
            '{:^5}| {:^5}'.format(
                round_totals[i],round_totals[i+1]
            ) for i in range(0, len(round_totals), 2)
        ]))

    print("-" * 74)
    # print("{:^9}{:^13}{:^13}{:^13}{:^13}{:^13}\n".format(
    #     "", "Win Rate:",
    #     *["{:.1f}%".format(100 * total_wins[a.player] / total_matches)
    #       for a in test_agents]
    # ))

    # print('{:^9}{:^13}'.format("", "Win Rate:") +
    #     ''.join([
    #         '{:^13}'.format(
    #             "{:.1f}%".format(100 * total_wins[x[1].player] / total_matches)
    #         ) for x in enumerate(test_agents)
    # ]))

    print('{:^9}{:^13}'.format("", "Win Rates:"))

    win_rates = []
    for x in enumerate(test_agents):
        rate = 100 * total_wins[x[1].player] / total_matches
        win_rates.append((x[1].name, rate))
        # print("{} - {:.1f}%".format(x[1].name, 100 * total_wins[x[1].player] / total_matches))

    for x in sorted(win_rates, reverse=True, key=lambda x: x[1]):
        print("{} - {:.1f}%".format(x[0], x[1]))

    if total_timeouts:
        print(("\nThere were {} timeouts during the tournament -- make sure " +
               "your agent handles search timeout correctly, and consider " +
               "increasing the timeout margin for your agent.\n").format(
            total_timeouts))
    if total_forfeits:
        print(("\nYour ID search forfeited {} games while there were still " +
               "legal moves available to play.\n").format(total_forfeits))

    for a, d in average_timeout_depths.items():
        try:
            print("{}-{}: {}".format(type(a), a.score, d))
        except AttributeError:
            print("{}: {}".format(type(a), d))

"""
Create some grid search agents
"""
# wed. night trials
# gs_constants = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
# specific_gs_constants = [[1, -1, 1, 0, 0],
# [1, 0, 1, 0, -1],
# [0, 0, 1, 1, 0],
# [0, 0, 1, 1, -1],
# [1, 1, 0, -1, 0],
# [1, 1, 0, -1, 1],
# [-1, -1, 0, -1, 0],
# [-1, 0, -1, 0, 0],
# [0, -1, 1, 0, -1],
# [1, -1, -1, 0, -1],
# [1, -1, -1, 0, 1],
# [1, -1, 0, 0, 0],
# [1, -1, 1, 0, -1],
# [1, 0, 1, 1, 0],
# [-1, 1, 0, 0, 0],
# [1, -1, 0, -1, -1],
# [1, 0, 1, 0, 1],
# [1, 0, 1, 1, -1],
# [-1, -1, 0, 1, 0],
# [0, -1, 1, 0, 0],
# [0, 0, 1, -1, -1],
# [1, -1, -1, 1, 1],
# [1, 0, -1, 0, 1]]

# fri overnight trial
# gs_constants = [[1, 2], #own_moves
#                 [-2, -1], #opp_moves
#                 [1, 2], #move_ratio
#                 [0], #own_openness
#                 [-1, 0], #opp_openness
#                 [0, 1], #openness_ratio
#                 [0, 1], #own_centerness
#                 [-1, 0], #opp_centerness
#                 [-1, 0, 1]] #centerness_ratio
# specific_gs_constants = [[2, -2, 1, 0, -1, 0, 1, -1, -1],
# [1, -1, 2, 0, -1, 0, 1, -1, -1],
# [1, -1, 1, 0, 0, 1, 0, -1, 1],
# [1, -2, 2, 0, 0, 0, 0, -1, -1],
# [2, -2, 1, 0, 0, 0, 0, 0, 0],
# [1, -2, 2, 0, -1, 0, 0, -1, 0],
# [1, -2, 2, 0, -1, 0, 0, -1, 1],
# [1, -1, 2, 0, -1, 0, 0, -1, 1],
# [2, -2, 1, 0, 0, 0, 1, 0, 0],
# [2, -2, 2, 0, -1, 0, 0, 0, 1],
# [2, -1, 1, 0, -1, 0, 1, 0, -1],
# [1, -2, 2, 0, 0, 0, 1, 0, -1],
# [1, -1, 2, 0, 0, 1, 0, -1, -1],
# [2, -2, 1, 0, -1, 1, 1, -1, -1],
# [2, -2, 1, 0, 0, 0, 0, -1, 1],
# [2, -2, 2, 0, -1, 0, 0, 0, -1],
# [2, -2, 2, 0, 0, 0, 0, 0, -1],
# [2, -2, 2, 0, 0, 0, 0, 0, 1],
# [2, -1, 1, 0, 0, 0, 0, 0, -1],
# [2, -1, 1, 0, 0, 0, 1, 0, 0],
# [2, -1, 2, 0, 0, 1, 1, -1, -1],
# [1, -2, 1, 0, -1, 0, 1, 0, -1],
# [1, -2, 1, 0, -1, 1, 1, 0, 0],
# [1, -2, 1, 0, 0, 0, 0, -1, 1],
# [1, -2, 2, 0, -1, 1, 0, -1, 0],
# [1, -2, 2, 0, -1, 1, 1, -1, -1],
# [1, -1, 1, 0, -1, 0, 1, -1, -1],
# [1, -1, 1, 0, 0, 0, 1, -1, -1],
# [1, -1, 2, 0, -1, 0, 0, 0, -1],
# [1, -1, 2, 0, -1, 1, 0, 0, 0],
# [1, -1, 2, 0, 0, 0, 0, -1, 0],
# [1, -1, 2, 0, 0, 0, 1, -1, 0],
# [1, -1, 2, 0, 0, 0, 1, 0, 0],
# [2, -2, 1, 0, -1, 0, 0, -1, -1],
# [2, -2, 1, 0, -1, 0, 0, 0, 1],
# [2, -2, 1, 0, -1, 0, 1, -1, 0],
# [2, -2, 1, 0, 0, 0, 0, 0, 1],
# [2, -2, 2, 0, -1, 0, 0, -1, 0],
# [2, -2, 2, 0, -1, 0, 1, 0, 0],
# [2, -2, 2, 0, -1, 1, 0, -1, 1],
# [2, -2, 2, 0, 0, 1, 1, 0, 0],
# [2, -1, 1, 0, -1, 0, 1, -1, 0],
# [2, -1, 1, 0, 0, 0, 1, 0, 1],
# [2, -1, 2, 0, -1, 0, 1, -1, 0],
# [2, -1, 2, 0, 0, 0, 1, -1, 0],
# [2, -1, 2, 0, 0, 1, 1, -1, 1],
# [1, -2, 1, 0, 0, 1, 0, -1, 0],
# [1, -2, 1, 0, 0, 1, 0, 0, 0],
# [1, -2, 2, 0, -1, 0, 0, 0, 1],
# [1, -2, 2, 0, -1, 0, 1, 0, -1],
# [1, -2, 2, 0, 0, 0, 1, 0, 1],
# [1, -1, 1, 0, 0, 0, 0, 0, -1],
# [1, -1, 1, 0, 0, 0, 1, -1, 1],
# [2, -2, 1, 0, -1, 0, 0, 0, 0],
# [2, -2, 1, 0, -1, 1, 1, 0, -1],
# [2, -2, 1, 0, 0, 0, 0, -1, 0],
# [2, -2, 2, 0, -1, 1, 0, -1, 0],
# [2, -2, 2, 0, -1, 1, 1, 0, -1],
# [2, -2, 2, 0, 0, 0, 0, -1, 1],
# [2, -2, 2, 0, 0, 1, 0, -1, -1],
# [2, -1, 1, 0, -1, 0, 0, -1, -1],
# [2, -1, 1, 0, -1, 0, 0, 0, 0],
# [2, -1, 1, 0, 0, 0, 1, -1, -1],
# [2, -1, 2, 0, -1, 1, 0, -1, 0],
# [2, -1, 2, 0, -1, 1, 1, -1, 0],
# [2, -1, 2, 0, -1, 1, 1, 0, 0],
# [2, -1, 2, 0, 0, 0, 0, -1, 0],
# [2, -1, 2, 0, 0, 0, 0, 0, -1],
# [1, -2, 1, 0, -1, 0, 0, 0, -1],
# [1, -2, 1, 0, -1, 0, 0, 0, 0],
# [1, -2, 1, 0, -1, 0, 0, 0, 1],
# [1, -2, 1, 0, -1, 1, 0, 0, 0],
# [1, -2, 1, 0, -1, 1, 1, 0, 1],
# [1, -2, 1, 0, 0, 0, 0, 0, -1],
# [1, -2, 1, 0, 0, 0, 1, -1, 0],
# [1, -2, 1, 0, 0, 0, 1, -1, 1],
# [1, -2, 2, 0, -1, 0, 0, 0, -1],
# [1, -2, 2, 0, -1, 1, 1, -1, 1],
# [1, -2, 2, 0, 0, 0, 0, 0, 1],
# [1, -2, 2, 0, 0, 1, 1, 0, 1],
# [1, -1, 1, 0, -1, 1, 1, 0, -1],
# [1, -1, 1, 0, -1, 1, 1, 0, 1],
# [1, -1, 1, 0, 0, 1, 1, -1, -1],
# [1, -1, 1, 0, 0, 1, 1, 0, -1],
# [1, -1, 1, 0, 0, 1, 1, 0, 1],
# [1, -1, 2, 0, -1, 0, 1, -1, 0],
# [1, -1, 2, 0, -1, 1, 0, 0, 1],
# [1, -1, 2, 0, 0, 0, 0, 0, 1],
# [1, -1, 2, 0, 0, 1, 0, -1, 1],
# [1, -1, 2, 0, 0, 1, 1, 0, 0],
# [2, -2, 1, 0, -1, 0, 0, -1, 1],
# [2, -2, 1, 0, -1, 0, 1, 0, 0],
# [2, -2, 1, 0, -1, 1, 0, 0, 0],
# [2, -2, 1, 0, 0, 0, 1, -1, -1],
# [2, -2, 2, 0, -1, 0, 1, -1, -1],
# [2, -2, 2, 0, -1, 0, 1, -1, 1],
# [2, -2, 2, 0, -1, 0, 1, 0, 1],
# [2, -2, 2, 0, 0, 0, 0, 0, 0],
# [2, -2, 2, 0, 0, 1, 0, -1, 0],
# [2, -2, 2, 0, 0, 1, 0, -1, 1],
# [2, -2, 2, 0, 0, 1, 1, -1, 0],
# [2, -1, 1, 0, -1, 0, 0, 0, -1],
# [2, -1, 1, 0, -1, 1, 0, 0, -1],
# [2, -1, 1, 0, 0, 0, 0, -1, 1],
# [2, -1, 1, 0, 0, 1, 0, 0, 0],
# [2, -1, 1, 0, 0, 1, 1, 0, 1],
# [2, -1, 2, 0, -1, 0, 0, -1, 1],
# [2, -1, 2, 0, -1, 0, 1, -1, -1],
# [2, -1, 2, 0, -1, 1, 0, -1, 1],
# [2, -1, 2, 0, -1, 1, 0, 0, -1],
# [2, -1, 2, 0, -1, 1, 0, 0, 1],
# [2, -1, 2, 0, 0, 0, 0, 0, 0],
# [2, -1, 2, 0, 0, 0, 1, -1, 1],
# [2, -1, 2, 0, 0, 0, 1, 0, 0],
# [2, -1, 2, 0, 0, 1, 1, 0, 1]]

# specific_gs_constants = [
# [2, -1, 2, 0, 0, 0, 0, -1, 0],
# [2, -1, 2, 0, 0, 1, 1, -1, 1],
# [2, -1, 1, 0, 0, 0, 1, 0, 0],
# [1, -2, 2, 0, 0, 0, 1, 0, 1],
# [2, -2, 1, 0, 0, 0, 0, -1, 0],
# [2, -1, 2, 0, 0, 0, 0, 0, -1],
# [2, -2, 2, 0, 0, 0, 0, 0, 0],
# [2, -2, 2, 0, 0, 1, 1, -1, 0],
# [1, -2, 2, 0, 0, 0, 0, -1, -1],
# [2, -2, 2, 0, 0, 0, 0, -1, 1],
# [2, -1, 1, 0, 0, 0, 1, -1, -1],
# [2, -1, 1, 0, 0, 1, 0, 0, 0],
# [2, -1, 2, 0, -1, 0, 1, -1, -1],
# [4, -1, 4, 0, 0, 0, 0, -1, 0],
# [4, -1, 4, 0, 0, 1, 1, -1, 1],
# [4, -1, 1, 0, 0, 0, 1, 0, 0],
# [1, -2, 4, 0, 0, 0, 1, 0, 1],
# [4, -2, 1, 0, 0, 0, 0, -1, 0],
# [4, -1, 4, 0, 0, 0, 0, 0, -1],
# [4, -2, 4, 0, 0, 0, 0, 0, 0],
# [4, -2, 4, 0, 0, 1, 1, -1, 0],
# [1, -2, 4, 0, 0, 0, 0, -1, -1],
# [4, -2, 4, 0, 0, 0, 0, -1, 1],
# [4, -1, 1, 0, 0, 0, 1, -1, -1],
# [4, -1, 1, 0, 0, 1, 0, 0, 0],
# [4, -1, 4, 0, -1, 0, 1, -1, -1]
# ]

# specific_gs_constants = [
# [4, -1, 4, 0, 0, 0, 0, 0, -1],
# [4, -1, 4, 0, 0, 0, 0, -1, 0],
# [4, -1, 1, 0, 0, 0, 1, -1, -1],
# [8, -1, 4, 0, 0, 0, 0, 0, -1],
# [8, -1, 4, 0, 0, 0, 0, -1, 0],
# [8, -1, 1, 0, 0, 0, 1, -1, -1],
# [8, -1, 8, 0, 0, 0, 0, 0, -1],
# [8, -1, 8, 0, 0, 0, 0, -1, 0],
# [8, -1, 1, 0, 0, 0, 1, -1, -1],
# [8, -1, 8, 0, 0, 0, 0, 0, -1],
# [8, -1, 8, 0, 0, 0, 0, -1, 0],
# [8, -1, 1, 0, 0, 0, 1, -1, -1],
# ]
#
# gs_funcs = {}
#
#
# # for i, c in enumerate(itertools.product(*gs_constants)):
# for i, c in enumerate(specific_gs_constants):
#     label = '_'.join([str(x) for x in list(c)])
#
#     exec("""def gs_score_func_{}(game, player):
#             return custom_score_general(game, player, {})""".format(i, c))
#
#     exec("""gs_funcs[label] = gs_score_func_{}""".format(i))
# gs2_funcs = {}
# gs2_constants = [[2], #own_moves
#                 [-2, -1], #opp_moves
#                 [2], #move_ratio
#                 [0.25, 0.5], #apply centerness based on completeness
#                 [-2, -1], #own_centerness
#                 [1, 2], #opp_centerness
#                 [-2, -1]] #centerness_ratio
#
# specific_gs2_constants = [
# [2, -1, 2, 0.5, -1, 1, -2],
# [2, -1, 2, 0.5, -2, 1, -2],
# [2, -1, 2, 0.5, -2, 2, -1],
# [2, -2, 2, 0.5, -1, 1, -1],
# [2, -1, 2, 0.25, -2, 2, -1],
# [2, -1, 2, 0.25, -1, 1, -1],
# [2, -1, 2, 0.25, -1, 2, -2],
# [2, -2, 2, 0.5, -2, 2, -1],
# [2, -1, 2, 0.25, -1, 2, -1],
# [2, -1, 2, 0.5, -1, 1, -1],
# [2, -2, 2, 0.25, -1, 2, -2],
# [2, -2, 2, 0.25, -1, 2, -1],
# [2, -1, 2, 0.25, -2, 1, -1],
# [2, -1, 2, 0.5, -1, 2, -2]
# ]

# for i, c in enumerate(itertools.product(*gs2_constants)):
# for i, c in enumerate(specific_gs2_constants):
#     label = '_'.join([str(x) for x in list(c)])
#
#     exec("""def gs2_score_func_{}(game, player):
#             return custom_score_general2(game, player, {})""".format(i, c))
#
#     exec("""gs2_funcs[label] = gs2_score_func_{}""".format(i))

def main():

    # Define two agents to compare -- these agents will play from the same
    # starting position against the same adversaries in the tournament
    test_agents = [
        Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved"),
        Agent(AlphaBetaPlayer(score_fn=custom_score), "AB_Custom"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_2), "AB_Custom_2"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_3), "AB_Custom_3")
    ]

    """
    Uncomment to add grid search function agents to test_agents
    """
    # print("{} grid search functions".format(len(gs_funcs)))
    # for name, func in gs_funcs.items():
    #     test_agents.append(Agent(AlphaBetaPlayer(score_fn=func), "AB_{}".format(name)))

    # print("{} grid search2 functions".format(len(gs2_funcs)))
    # for name, func in gs2_funcs.items():
    #     test_agents.append(Agent(AlphaBetaPlayer(score_fn=func), "AB_GS2_{}".format(name)))

    # Define a collection of agents to compete against the test agents
    cpu_agents = [
        Agent(RandomPlayer(), "Random"),
        Agent(MinimaxPlayer(score_fn=open_move_score), "MM_Open"),
        Agent(MinimaxPlayer(score_fn=center_score), "MM_Center"),
        Agent(MinimaxPlayer(score_fn=improved_score), "MM_Improved"),
        Agent(AlphaBetaPlayer(score_fn=open_move_score), "AB_Open"),
        Agent(AlphaBetaPlayer(score_fn=center_score), "AB_Center"),
        Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved")
    ]

    print(DESCRIPTION)
    print("{:^74}".format("*************************"))
    print("{:^74}".format("Playing Matches"))
    print("{:^74}".format("*************************"))
    play_matches(cpu_agents, test_agents, NUM_MATCHES)


if __name__ == "__main__":
    main()
