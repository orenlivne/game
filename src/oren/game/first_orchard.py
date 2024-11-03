#!/usr/bin/env python
"""
First Orchard game analyis. We figure out the winning probability.

Game Rules (See https://boardgamegeek.com/thread/3185262/first-orchard-review-rolls-in-the-family):
* On your turn, you roll the die, and get to pick the matching fruit off of a tree and put it in the basket. 
* If you roll a basket, you can choose any fruit. 
However, if you roll a raven, then the black raven will move one step along the path, inching closer to the fruit
* that remains on the trees. 
* If you can collect all of the fruit before the raven reaches the end of the path, you win, otherwise you lose.

This a standard application of calculating absorbing probabilities in an absorbing Markov chain; see
https://en.wikipedia.org/wiki/Absorbing_Markov_chain .

Our result is that the best strategy seems to be to pick a random tree if you land on a basket die value. The seemingly
best strategy (choosing the tree with the maximum fruit) turns out to be the worst. This seems counterintuitive as it
should waste less turns towards the end of the game, but I guess not!
"""
# TODO:
# - Dependence on n, T, R.
# - Confirm by simulating the game: DONE.
# - How does result sit with #turns for raven to each 5 ~ 5 * 6 = 30; # turns to clear all fruit ~ 16 * (6/5) = 19.2
# except for wasted turns on empty trees, so winning probability seems a lot higher (more than 50%). Answer: there are
# a lot of wasted turns towards the end of the game when trees are nearly empty.
# - Effect of strategy: looks like RANDOM > MIN_TREE > MAX_TREE. Why?

import argparse
import itertools
import numpy as np
import scipy.sparse as scs
import time
from enum import Enum
from typing import Tuple, List
from scipy.sparse.linalg import spsolve


# Absorbing state indices.
IN_PROGRESS = 2
LOSS, WIN = range(IN_PROGRESS)

class Strategy(Enum):
    """Basket strategy."""
    # Strategy A: Pick a fruit from the tree with the most fruit to minimize
    # the chance of an empty tree in rule 1, which would mean a wasted turn.
    MAX_TREE = 0
    # Strategy B: choose the tree with the least fruits. Seems like the worst strategy but turns out to be the best.
    MIN_TREE = 1
    # Random tree with fruits.
    RANDOM = 2


class FirstOrchardSimulator:
    def __init__(self, n: int, t_max: int, r_max: int, strategy: Strategy, rng: np.random.Generator) -> int:
        self._n = n
        self._t_max = t_max
        self._r_max = r_max
        self._strategy = strategy
        self._rng = rng
        self._state = None

    def play(self):
        p = 1 / (self._n + 2)
        # Initialize game state.
        self._state = np.array((self._t_max, ) * self._n + (0, ), dtype=int)
        result = self._result
        #print(self._state)
        while result == IN_PROGRESS:
            die_value = rng.integers(self._n + 2)
            self._update(die_value)
            #print(die_value, self._state)
            result = self._result
        return result
        
    def _update(self, die_value: int) -> None:
        """Updates the game state given that the die landed on the value `die_value`."""
        n, s = self._n, self._state
        if die_value <= n:
            # Rule 1: a fruit is transferred from a tree to the basket.
            if die_value == n:
                # Rule 2: Basket.
                if self._strategy == Strategy.MAX_TREE:
                    # Strategy A: we assume the strategy is to pick a fruit from the tree with the most fruit to minimize
                    # the chance of an empty tree in rule 1, which would mean a wasted turn.
                    i = np.argmax(s[-1])
                elif self._strategy == Strategy.MIN_TREE:
                    # Strategy B: choose the tree with the least fruits.
                    nonempty_tree_idx = np.argwhere(s[:-1] > 0)
                    i = nonempty_tree_idx[np.argmin(s[nonempty_tree_idx])][0]
                elif self._strategy == Strategy.RANDOM:
                    # Strategy C: choose a random tree with fruits.
                    nonempty_tree_idx = np.argwhere(s[:-1] > 0).flatten()
                    i = np.random.choice(nonempty_tree_idx)
                else:
                    raise ValueError(f"Invalid strategy value {self._strategy}")
            else:
                i = die_value
            if s[i] > 0:
                s[i] -= 1
        else:
            # Rule 3: raven advances.
            s[-1] += 1

    @property
    def _result(self) -> int:
        state = self._state
        if state[-1] == self._r_max:
            if max(state[:-1]) == 0:
                raise ValueError("Invalid game state: trees are empty and raven in Orchard.")
            # Raven reached orchard.
            return LOSS
        if max(state[:-1]) == 0:
            # All trees are empty, it's a win.
            return WIN
        # Still in progress.
        return IN_PROGRESS


class FirstOrchardAnalyzer:
    def __init__(self, n: int, t_max: int, r_max: int, strategy: Strategy) -> int:
        self._n = n
        self._t_max = t_max
        self._r_max = r_max
        self._strategy = strategy
        # State is encoded as the vector (t[0],...,t[n-1],r) where t[i] = #fruits in tree i and r = raven state.
        # Map each state vector 'state' into a 1D index 'idx'. We call this mapping s.
        self._s = dict((state, idx) for idx, state in enumerate(self._all_states_iter()))

    @property
    def initial_state(self) -> int:
        return self._s[(self._t_max, ) * self._n + (0, )]
    
    def _all_states_iter(self):
        shape = (self._t_max + 1, ) * self._n + (self._r_max + 1, )
        return itertools.product(*(range(state_dim_size) for state_dim_size in shape))

    def transition_matrices(self) -> Tuple[scs.csc_matrix, scs.csc_matrix]:
        n, r_max, s = self._n, self._r_max, self._s
        # Assuming a fair (n+2)-sided dice.
        p = 1 / (n + 2)
        # nts = number of transient states.
        nts = len(self._s)
        # Create an edge list for Q, A: (from state, to state, transition probability) based on the game rules.
        # TODO: pre-allocate list for effiency.
        q_edges = []
        a_edges = []
        for idx, state in enumerate(self._all_states_iter()):
            #print(idx, state)
            if max(state[:-1]) > 0 and state[-1] < r_max:
                # Transient game state.
                # Rule 1: a fruit is transferred from a tree to the basket.
                for i in range(n):
                    q_edges.append((s[state], s[state[:i] + (max(0, state[i] - 1), ) + state[i+1:]], p))
    #                print("\t", "Q", s[state], s[state[:i] + (max(0, state[i] - 1), ) + state[i+1:]], p)
                # Rule 2: Basket: we assume the strategy is to pick a fruit from the tree with the most fruit to minimize
                # the chance of an empty tree in rule 1, which would mean a wasted turn.
                # Rule 2: Basket.
                if self._strategy == Strategy.MAX_TREE:
                    # Strategy A: we assume the strategy is to pick a fruit from the tree with the most fruit to minimize
                    # the chance of an empty tree in rule 1, which would mean a wasted turn.
                    i = np.argmax(state[-1])
                elif self._strategy == Strategy.MIN_TREE:
                    # Strategy B: choose the tree with the least fruits.
                    state_array = np.array(state, dtype=int)
                    nonempty_tree_idx = np.argwhere(state_array[:-1] > 0)
                    i = nonempty_tree_idx[np.argmin(state_array[nonempty_tree_idx])][0]
                elif self._strategy == Strategy.RANDOM:
                    # Strategy C: choose a random tree with fruits. In this analysis mode, this is only an approximation;
                    # one should really average over different random graphs instead of generating just one, but it looks
                    # like even one with random edges fits the experimental result quite nicely.
                    state_array = np.array(state, dtype=int)
                    nonempty_tree_idx = np.argwhere(state_array[:-1] > 0).flatten()
                    i = np.random.choice(nonempty_tree_idx)
                else:
                    raise ValueError(f"Invalid strategy value {self._strategy}")
                q_edges.append((s[state], s[state[:i] + (max(0, state[i] - 1), ) + state[i+1:]], p))
                # Rule 3: raven advances.
                q_edges.append((s[state], s[state[:-1] + (state[-1] + 1, )], p))
            elif state[-1] == r_max:
                # Raven reaches orchard, it's a loss.
                a_edges.append((s[state], LOSS, 1))
    #            print("\t", "A", s[state], LOSS, 1)
            elif max(state[:-1]) == 0:
                # All trees are empty, it's a win.
                a_edges.append((s[state], WIN, 1))
    #            print("\t", "A", s[state], WIN, 1)
       # print(len(q_edges), len(a_edges))
        q = to_sparse_matrix(q_edges, (nts, nts))
        a = to_sparse_matrix(a_edges, (nts, 2))
        return q, a


def parse_args():
    """Defines and parses command-line flags."""
    parser = argparse.ArgumentParser(description="Nerdle Solver.")
    parser.add_argument("--n", type=int, default=4, required=False, help="Number of trees.")
    parser.add_argument("--t", type=int, default=4, required=False, help="Initial number of fruits per tree.")
    parser.add_argument("--r", type=int, default=5, required=False, help="Number of raven steps required to lose the game.")
    return parser.parse_args()


def to_sparse_matrix(edge_list: List[Tuple[int, int, float]], shape: Tuple[int]) -> scs.csc_matrix:
    row_ind, col_ind, data = zip(*edge_list)
    return scs.csc_matrix((data, (row_ind, col_ind)), shape=shape)


if __name__ == "__main__":
    args = parse_args()

    for strategy in Strategy:
        print(f"strategy {strategy}")

        # Run game.
        rng = np.random.default_rng(seed=int(time.time()))
        simulator = FirstOrchardSimulator(args.n, args.t, args.r, strategy, rng)
        num_simulations = 10000
        result = np.array([simulator.play() for _ in range(num_simulations)])
        print(f"Experimental probability: {np.bincount(result)[WIN] / num_simulations:.3f}")
        
        # Analysis.
        rng = np.random.default_rng(seed=int(time.time()))
        analyzer = FirstOrchardAnalyzer(args.n, args.t, args.r, strategy)
        q, a = analyzer.transition_matrices()
        # p  = scs.hstack((q, a))
        # print(max(abs(np.squeeze(np.array(p.sum(axis=1)) - 1))))
        b = spsolve(scs.eye(q.shape[0]).tocsc() - q, a)
    #    print(max(abs(np.squeeze(np.array(b.sum(axis=1)) - 1))))
        print(f"Theoretical probability : {b[analyzer.initial_state, WIN]:.3f}")
