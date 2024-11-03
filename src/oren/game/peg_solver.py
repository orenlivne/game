#!/usr/bin/env python
"""Loyd Peg White/Black game solver. The goal is to minimize the number of moves to swap the locations of the White and
Black pegs."""
import functools
import itertools
import networkx as nx
from typing import Tuple, List


class Board:
    def __init__(self, size=3):
        self._board = self._board_matrix(size)
        self._shape = len(self._board), len(self._board[0])

    def __repr__(self):
        return "\n".join(
            (" ".join(map(lambda x: "%3s" % ("." if x < 0 else str(x)), row[1:-1])))
            for row in self._board[1:-1])

    def nbhr(self):
        num_squares = max(max(row) for row in self._board) + 1
        sub_of = [None] * num_squares
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                if self._board[i][j] >= 0:
                    sub_of[self._board[i][j]] = (i, j)
        return [[self._board[i_nbhr][j_nbhr] for i_nbhr, j_nbhr in self._neighbors(sub_of[k])]
                for k in range(num_squares)]

    def print_path(self, path):
        print(self._state_repr(path[0]))
        for i, state in enumerate(path[1:], 1):
            prev_state = path[i - 1]
            peg_color = prev_state[1] & (1 << state[0])
            print("{}. Move {} peg from {} to {}".format(i, "white" if peg_color else "black", state[0], prev_state[0]))
            print(self._state_repr(state))

    def _state_repr(self, state):
        return "\n".join(
            (" ".join(map(lambda x: "%3s" % ("." if x < 0 else
                                             ("*" if x == state[0] else "W" if state[1] & (1 << x) else "B")),
                          row[1:-1])))
            for row in self._board[1:-1])

    @staticmethod
    def _board_matrix(size) -> List:
        """Returns the board: an array with all squares sequentially numbered. Padded for easy neighbor calculation."""
        padded_size = 2 * size + 1
        m = [[-1] * padded_size for _ in range(padded_size)]
        # First board half.
        for i in range(size):
            for j in range(size):
                m[i + 1][j + 1] = size * i + j
        # Second board half. Overlaps the first board at its top left corner.
        offset = size ** 2 - 1
        for i in range(size):
            for j in range(size):
                m[i + size][j + size] = size * i + j + offset
        return m

    def _neighbors(self, sub):
        i, j = sub
        m, n = self._shape
        legal_moves = [
            (i-2, j), (i-1, j), (i+1, j), (i+2, j),
            (i, j-2), (i, j-1), (i, j+1), (i, j+2)
        ]
        return filter(lambda x: 0 <= x[0] < m and 0 <= x[1] < n and self._board[x[0]][x[1]] >= 0, legal_moves)


def swap_bits(n, p1, p2):
    """Swaps bit at positions p1 and p2 in an integer n."""
    # Move p1'th to rightmost side.
    bit1 = (n >> p1) & 1
    # Move p2'th to rightmost side.
    bit2 = (n >> p2) & 1
    # XOR the two bits.
    x = (bit1 ^ bit2)
    # Put the xor bit back to their original positions.
    x = (x << p1) | (x << p2)
    # XOR 'x' with the original number so that the two sets are swapped.
    result = n ^ x
    return result


def insert_zero_bit(x, p):
    return (x >> p) << (p + 1) | (x & (1 << p) - 1)


def bit_locations_to_bin(locations):
    return functools.reduce(lambda x, y: x | y, ((1 << y) for y in locations))


def create_state_graph(nbhr):
    num_squares = len(nbhr)
    num_white_pegs = num_squares // 2
    peg_states = list(itertools.combinations(range(num_squares - 1), num_white_pegs))
    print("# peg_states {}".format(len(peg_states)))
    states = [(empty_location, insert_zero_bit(bit_locations_to_bin(locations), empty_location))
              for empty_location in range(num_squares)
              for locations in peg_states]
    print("# states {}".format(len(states)))
    edges = (((empty_location, peg_state), (peg_location, swap_bits(peg_state, empty_location, peg_location)))
             for empty_location, peg_state in states
             for peg_location in nbhr[empty_location])
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


if __name__ == "__main__":
    size = 3
    board = Board(size)
    print("Board:")
    print(board)
    nbhr = board.nbhr()
    print("nbhr array len {}".format(len(nbhr)))

    g = create_state_graph(board.nbhr())
    print("Built graph", g.number_of_nodes(), g.number_of_edges())

    locations = list(range(size ** 2 - 1))
    empty_location = size ** 2 - 1
    source = (empty_location, insert_zero_bit(bit_locations_to_bin(locations), empty_location))

    locations = list(range(size ** 2 - 1, 2 * size ** 2 - 2))
    empty_location = size ** 2 - 1
    target = (empty_location, insert_zero_bit(bit_locations_to_bin(locations), empty_location))

    path = nx.shortest_path(g, source=source, target=target)

    print("Minimum number of moves: {}".format(len(path) - 1))
    board.print_path(path)

    all_paths = nx.all_shortest_paths(g, source=source, target=target)
    print("Number of optimal solutions: {}".format(sum(1 for _ in all_paths)))
