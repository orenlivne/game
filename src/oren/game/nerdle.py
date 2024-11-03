#!/usr/bin/env python
"""
Nerdle solver. A straightforward application of Knuth's algorithm as applied to Mastermind and implemented in
Python in https://betterprogramming.pub/solving-mastermind-641411708d01.

"Nerdle answers don't use leading zeros or lone zeros, even though some may be accepted as valid guesses.
So you won't find an answer like this: 0+5+5=10, or like this: 01+2+1=4. However, 0 is allowed as the
answer to the right of the equal sign (i.e. 5-3-2=0 is allowed)."
Thus:
1. = must be in th e5th, 6th, or 7th slots in the answer.
2. (= can be in 2nd..7th slot in a guess.).

"Nerdle answers don't start with negative numbers or have a negative number after the equals sign, even though
these may be accepted as valid guesses. So you won't find an answer like this: -5-6=-11. We get it."

3. Only digits may follow =.
4. An operation must be followed by a number.
5. The first slot must be a digit.
6. There has to be at least one op, since the number of slots (8) is even, so can't have xxx=xxx as in 7 slots, say.

Repeating numbers:
https://nerdschalk.com/can-nerdle-repeat-numbers-and-symbols-same-number-twice-rule-explained/

References:
    1. https://betterprogramming.pub/solving-mastermind-641411708d01
    2. https://github.com/aydinschwa/Mastermind-AI
    3. https://www.youtube.com/watch?v=Okm_t5T1PiA&ab_channel=Confreaks
    4. https://www.cs.uni.edu/~wallingf/teaching/cs3530/resources/knuth-mastermind.pdf
"""
import argparse
import collections
import itertools
import os
import pickle
from enum import Enum
from typing import Tuple, List


def parse_args():
    """Defines and parses command-line flags."""
    parser = argparse.ArgumentParser(description="Nerdle Solver.")
    parser.add_argument("--mode", default=None, required=True, choices=["build_dict"],
                        help="Running mode. 'build_dict': builds score dictionary and saves to a file.")
    parser.add_argument("--slots", default=6, type=int, help="Number of slots in answer.")
    parser.add_argument("--score_dict_file", default="score_dict.pickle",
                        help="Path to score dictionary pickle file name.")
    return parser.parse_args()


class Operator(Enum):
    """Binary arithmetic operators allowed in a Nerdle expression."""
    PLUS = "+"
    MINUS = "-"
    TIMES = "*"
    DIVIDE = "/"


class Hint(Enum):
    """Hint characters."""
    INCORRECT = 0       # Nerdle black
    CORRECT = 1         # Nerdle green
    MISPLACED = 2       # Nerdle purple


def diff(x: Tuple[int]) -> Tuple[int]:
    return tuple(x[i + 1] - x[i] for i in range(len(x) - 1))


def all_answers(num_slots: int) -> List[str]:
    """Generates all possible Nerdle answers of size 'num_slots'."""
    # TODO: prune the combinations we loop over.
    # TODO: use direct evaluation instead of eval().
    for num_param in range(3, num_slots - 1):
        num_result_slots = num_slots - num_param - 1
        result_range = (0 if num_result_slots == 1 else 10 ** (num_result_slots - 1), 10 ** num_result_slots)
        print("param_slots", num_param, "X" * num_param + " = " + "X" * num_result_slots,
              "result_range", result_range)
        for num_ops in range(1, (num_param - 1) // 2 + 1):
            #print("\t", "num_ops", num_ops)
            op_slots = [
                combination
                for combination in itertools.combinations(range(1, num_param - 1), num_ops)
                if len(combination) == 1 or min(diff(combination)) > 1
                ]
            #print("\t", op_slots)
            for op_slot in op_slots:
                param_lens = [(n - 1) for n in diff((-1, ) + op_slot + (num_param, ))]
                #print("\t\t", "op_slot", op_slot, "param_lens", param_lens)
                print("\t\t", "o".join("X" * n for n in param_lens) + " = " + "X" * (num_slots - num_param - 1))
                for param_values in itertools.product(*(
                        list(itertools.chain.from_iterable(
                            tuple((range(10 ** (n - 1), 10 ** n), "+-*/") for n in param_lens)))[:-1]
                )):
                    s = "".join(map(str, param_values))
                    result = eval(s)
                    if result_range[0] <= result < result_range[1] and \
                            (isinstance(result, int) or result.is_integer()):
                        yield param_values, int(result)


def score_guess(guess: str, answer: str) -> int:
    """
    Returns the score of a guess.

    :param guess: Guess string.
    :param answer: Answer string.
    :return: Hint string, coded as a binary number. First 2 LSBs = first slot hint, etc.
    """
    # Coded below uses the assumptions that INCORRECT=0 and there are 2 bits of feedback per hint.

    # iterates through guess and answer lists element-by-element. Whenever it finds a match,
    # removes the value from a copy of answer so that nothing is double counted.
    hints = 0
    answer_no_match = []
    guess_no_match = []
    idx_no_match = []  # Indices of 'guess_no_match' characters.
    for idx, guess_elem, ans_elem in zip(range(len(guess)), guess, answer):
        if guess_elem == ans_elem:
            hints |= (Hint.CORRECT.value << (2 * idx))
        else:
            guess_no_match.append(guess_elem)
            answer_no_match.append(ans_elem)
            idx_no_match.append(idx)

    # Misplaced characters are flagged left-to-right, i.e., if there are two misplaced "1"s in the guess and one
    # "1" in the answer, the first "1" in the guess will be misplaced, the second incorrect.
    for idx, guess_elem in zip(idx_no_match, guess_no_match):
        if guess_elem in answer_no_match:
            hints |= (Hint.MISPLACED.value << (2 * idx))
            answer_no_match.remove(guess_elem)

    return hints


def build_dict(all_guesses):
    # default dict avoids storing keys as tuple, saves lookup time
    score_dict = collections.defaultdict(dict)
    n = len(all_guesses) ** 2
    for i, (guess, answer) in enumerate(itertools.product(all_guesses, repeat=2)):
        if i % 10000000 == 0:
            print("{} / {} ({:.1f}%) completed".format(i, n, (100 * i) / n))
        score_dict[guess][answer] = score_guess(guess, answer)
    return score_dict


class NerdleData:
    """Encapsulates data structures required for the solver."""
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.answers = set(all_answers(self.num_slots))
        self.score_dict = build_dict(self.answers)

    def __getstate__(self):
        return {"num_slots": self.num_slots, "answers": self.answers, "score_dict": self.score_dict}

    def __setstate__(self, d):
        self.num_slots = d["num_slots"]
        self.answers = d["answers"]
        self.score_dict = d["score_dict"]


class NerdleSolver:
    """
    Solves Nerdle queries.

    Note: modifies 'data' during solve() calls and then restores it. So cannot be used concurrently with
    the same 'data' object. Copying the data structures to each solver instance is too time and memory intensive.
    """
    def __init__(self, data: NerdleData, answer: str, max_guesses: int = 6):
        self._data = data
        self.all_answers = data.answers.copy()
        self.max_guesses = max_guesses
        self.answer = answer
        # Keeps track of the data.score_dict entries we modify in a solve() call. Typically, there are only few
        # of them.
        self._score_dict_backup = {}

    def make_guess(self, all_answers):
        guesses_to_try = []
        for guess, scores_by_answer_dict in self._data.score_dict.items():
            # Reduce possible_score_dict to only include possible answers.
            self._score_dict_backup[guess] = self._data_score_dict[guess]
            scores_by_answer_dict = {answer: score for answer, score in scores_by_answer_dict.items()
                                     if answer in all_answers}
            self._data.score_dict[guess] = scores_by_answer_dict

            # find how often a score appears in scores_by_answer_dict, get max
            possibilities_per_score = collections.Counter(scores_by_answer_dict.values())
            worst_case = max(possibilities_per_score.values())

            # prefer possible guesses over impossible ones
            impossible_guess = guess not in all_answers

            guesses_to_try.append((worst_case, impossible_guess, guess))

        return min(guesses_to_try)[-1]

    def restore_data(self):
        for guess, entry in self._score_dict_backup.items():
            self._data.score_dict[guess] = entry

    def solve(self, initial_guess: str = "0+12/3=4"):
        guesses_left = self.max_guesses
        all_answers = self._data.answers.copy()
        guess_history = []
        hint_history = []
        while guesses_left > 0:
            if guesses_left == self.max_guesses:
                guess = initial_guess
            else:
                guess = self.make_guess()
            guess_history.append(guess)

            # reduce amount of possible answers by checking answer against guess and score
            score = score_guess(guess, self.answer)
            hint_history.append(score)
            self.all_answers = {answer for answer in self.all_answers
                                if self._data.score_dict[guess][answer] == score}

            guesses_left -= 1
            if guess == self.answer:
                self.restore_data()
                return guess_history, hint_history

        # Failed to solve within the allotted number of guesses.
        self.restore_data()
        return None, None


if __name__ == "__main__":
    args = parse_args()

    # Create/load solver data.
    if os.path.exists(args.score_dict_file):
        with open(args.score_dict_file, "rb") as f:
            solver_data = pickle.load(f)
    else:
        solver_data = NerdleData(args.slots)
        with open(args.score_dict_file, "wb") as f:
            pickle.dump(solver_data, f)

    # if args.mode == "build_dict":
    # elif args.mode == "run":
    print("Slots {} answers {} score_dict size {}".format(
        solver_data.num_slots, len(solver_data.answers), len(solver_data.score_dict.items())))
#
#     for param_values, result in answers:
#         print(param_values, result)
# #        print("".join(map(str, param_values)) + "=" + str(int(result)))

    # Assumes that the set of all guesses = set of all answers. A Nerdle guess must "compute", i.e., must be a valid
    # arithmetic expression of the form 'LHS=RHS'. Since we assume the user knows the rules and does not enter unary
    # '-' operations of numbers with leading zeros, for instance, the set of guesses should equal the set of answers.
