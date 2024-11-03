"""Pawn game solver unit tests."""
import functools
import pytest
import game.nerdle.nerdle as nerdle
from game.nerdle.nerdle import Hint, NerdleData

SCORE_DICT_FILE = "score_dict.pickle"


@pytest.fixture
def solver_data():
    # Create/load solver data.
    if os.path.exists(SCORE_DICT_FILE):
        with open(SCORE_DICT_FILE, "rb") as f:
            solver_data = pickle.load(f)
    else:
        solver_data = NerdleData(args.slots)
        with open(SCORE_DICT_FILE, "wb") as f:
            pickle.dump(solver_data, f)
    return solver_data


class TestNerdle:
    def test_score(self):
        assert nerdle.score_guess("10-43=66", "12+34=56") == \
               hints_to_score((Hint.CORRECT, Hint.INCORRECT, Hint.INCORRECT, Hint.MISPLACED,
                               Hint.MISPLACED, Hint.CORRECT, Hint.INCORRECT, Hint.CORRECT))

        # Repeated digit. First occurrence is correct.
        assert nerdle.score_guess("10-84=46", "12+34=56") == \
               hints_to_score((Hint.CORRECT, Hint.INCORRECT, Hint.INCORRECT, Hint.INCORRECT,
                               Hint.CORRECT, Hint.CORRECT, Hint.INCORRECT, Hint.CORRECT))

        # Repeated digit. First occurrence is misplaced.
        assert nerdle.score_guess("10-43=46", "12+34=56") == \
               hints_to_score((Hint.CORRECT, Hint.INCORRECT, Hint.INCORRECT, Hint.MISPLACED,
                               Hint.MISPLACED, Hint.CORRECT, Hint.INCORRECT, Hint.CORRECT))

        # Repeated digit where second occurrence is the correct one. First one should be incorrect then.
        assert nerdle.score_guess("40-84=77", "12+34=56") == \
               hints_to_score((Hint.INCORRECT, Hint.INCORRECT, Hint.INCORRECT, Hint.INCORRECT,
                               Hint.CORRECT, Hint.CORRECT, Hint.INCORRECT, Hint.INCORRECT))

    def test_generate_all_answers(self, solver_data):
        assert all(len("".join(map(str, param_values)) + "=" + str(int(result))) == num_slots
                   for param_values, result in solver_data.answers)

    def test_solve(self, solver_data):
        solver = nerdle.NerdleSolver(solver_data, "12+34=46")
        guess_history, hint_history = solver.solve()
        assert len(guess_history) == 6


def hints_to_score(hints):
    return functools.reduce(lambda x, y: x | y, (hint.value << (2 * idx) for idx, hint in enumerate(hints)), 0)

