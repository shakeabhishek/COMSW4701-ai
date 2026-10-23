# Name: Abhishek Pillai
# UNI:  arp2247
#
# COMS 4701 - Homework 2 (Coding): 2048 adversarial search agent.
#
# Search : expectiminimax with alpha-beta pruning

import random
import time

from BaseAI import BaseAI

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------
TIME_BUDGET = 0.12          # CPU seconds per move (hard limit is 0.2 + 0.05)
CHECK_EVERY = 64            # nodes between deadline checks
TT_LIMIT    = 1500000       # transposition table size cap

P2, P4 = 0.9, 0.1           # tile spawn probabilities

LOST        = -1e9          # value of a board with no legal move for us
W_EMPTY     = 270.0
W_MERGES    = 700.0
W_MONO      = 47.0
MONO_POWER  = 4.0
W_SUM       = 11.0
SUM_POWER   = 3.5
W_BASE      = 200000.0

# ----------------------------------------------------------------------------
# Row tables, built once at import (before the game clock starts)
# ----------------------------------------------------------------------------
_ROW_MASK = 0xFFFF
_MOVE_L = [0] * 65536
_MOVE_R = [0] * 65536
_HEUR   = [0.0] * 65536


def _build_tables():
    pw_sum  = [r ** SUM_POWER if r else 0.0 for r in range(16)]
    pw_mono = [r ** MONO_POWER for r in range(16)]

    for row in range(65536):
        line = [(row >> (4 * i)) & 0xF for i in range(4)]

        # ---- heuristic score of this line ----
        total = 0.0
        empty = 0
        merges = 0
        prev = 0
        counter = 0
        for rank in line:
            total += pw_sum[rank]
            if rank == 0:
                empty += 1
            else:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter
                    counter = 0
                prev = rank
        if counter > 0:
            merges += 1 + counter

        mono_l = mono_r = 0.0
        for i in range(1, 4):
            if line[i - 1] > line[i]:
                mono_l += pw_mono[line[i - 1]] - pw_mono[line[i]]
            else:
                mono_r += pw_mono[line[i]] - pw_mono[line[i - 1]]

        _HEUR[row] = (W_BASE + W_EMPTY * empty + W_MERGES * merges
                      - W_MONO * min(mono_l, mono_r) - W_SUM * total)

        # ---- slide/merge toward index 0 (same rule as Grid.merge) ----
        tiles = [r for r in line if r]
        out = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                out.append(min(tiles[i] + 1, 15))
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        out += [0] * (4 - len(out))
        left = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)
        _MOVE_L[row] = left

        rev = line[::-1]
        tiles = [r for r in rev if r]
        out = []
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                out.append(min(tiles[i] + 1, 15))
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        out += [0] * (4 - len(out))
        out.reverse()
        _MOVE_R[row] = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)


_build_tables()


# ----------------------------------------------------------------------------
# Bitboard primitives
# ----------------------------------------------------------------------------
def _transpose(x):
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _left(b):
    L = _MOVE_L
    return (L[b & 0xFFFF] | (L[(b >> 16) & 0xFFFF] << 16)
            | (L[(b >> 32) & 0xFFFF] << 32) | (L[b >> 48] << 48))


def _right(b):
    R = _MOVE_R
    return (R[b & 0xFFFF] | (R[(b >> 16) & 0xFFFF] << 16)
            | (R[(b >> 32) & 0xFFFF] << 32) | (R[b >> 48] << 48))


def _up(b):
    return _transpose(_left(_transpose(b)))


def _down(b):
    return _transpose(_right(_transpose(b)))


# index = game move code: 0 UP, 1 DOWN, 2 LEFT, 3 RIGHT
_MOVES = (_up, _down, _left, _right)


def _evaluate(b):
    H = _HEUR
    t = _transpose(b)
    return (H[b & 0xFFFF] + H[(b >> 16) & 0xFFFF] + H[(b >> 32) & 0xFFFF] + H[b >> 48]
            + H[t & 0xFFFF] + H[(t >> 16) & 0xFFFF] + H[(t >> 32) & 0xFFFF] + H[t >> 48])


def _pack(grid):
    b = 0
    shift = 0
    for row in grid.map:
        for v in row:
            if v:
                b |= (v.bit_length() - 1) << shift
            shift += 4
    return b


class _Timeout(Exception):
    pass


# ----------------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------------
class IntelligentAgent(BaseAI):
    def __init__(self):
        self.tt = {}
        self.nodes = 0
        self.deadline = 0.0

    # -- search ---------------------------------------------------------------
    def _max_value(self, b, depth, beta):
        """MAX node. beta comes from the MIN node directly above (inf at root).
        Returns the exact value, or a lower bound >= beta if it was cut off."""
        if depth == 0:
            return _evaluate(b)

        key = (b, depth)
        hit = self.tt.get(key)
        if hit is not None:
            return hit

        self.nodes += 1
        if self.nodes % CHECK_EVERY == 0 and time.process_time() > self.deadline:
            raise _Timeout()

        alpha = LOST
        pruned = False
        for mv in _MOVES:
            nb = mv(b)
            if nb == b:
                continue
            v = self._chance_value(nb, depth)
            if v > alpha:
                alpha = v
                if alpha >= beta:
                    pruned = True
                    break

        if not pruned:
            if len(self.tt) > TT_LIMIT:
                self.tt.clear()
            self.tt[key] = alpha
        return alpha

    def _chance_value(self, b, depth):
        """CHANCE node: tile is a 2 (0.9) or a 4 (0.1); no pruning here."""
        cells = [s for s in range(0, 64, 4) if not (b >> s) & 0xF]
        if not cells:
            return self._max_value(b, depth - 1, float('inf'))  # unreachable after a move

        total = 0.0
        for tile, p in ((1, P2), (2, P4)):
            beta = float('inf')
            for s in cells:                       # MIN node: computer picks the cell
                v = self._max_value(b | (tile << s), depth - 1, beta)
                if v < beta:
                    beta = v
            total += p * beta
        return total

    def _search_root(self, b, depth, order):
        best_mv, best_val = None, LOST - 1
        for mv_idx in order:
            nb = _MOVES[mv_idx](b)
            if nb == b:
                continue
            v = self._chance_value(nb, depth)
            if v > best_val:
                best_val, best_mv = v, mv_idx
        return best_mv

    # -- entry point -----------------------------------------------------------
    def getMove(self, grid):
        self.deadline = time.process_time() + TIME_BUDGET
        self.nodes = 0

        b = _pack(grid)
        legal = [i for i in range(4) if _MOVES[i](b) != b]
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best = legal[0]
        order = legal
        depth = 1
        try:
            while depth <= 64:
                mv = self._search_root(b, depth, order)
                if mv is not None:
                    best = mv
                    order = [mv] + [i for i in legal if i != mv]
                depth += 1
        except _Timeout:
            pass
        return best
