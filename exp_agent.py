# Experimental, configurable copy of the agent used only for benchmarking
# (not part of the submission). configure() rebuilds the row tables.

import math
import time

from BaseAI import BaseAI

CFG = dict(
    heur='nn',          # 'nn' (per-row monotonic/merge/sum) or 'ov' (ovolve-style)
    budget=0.14,
    min_k=0,            # 0 = MIN tries every empty cell; k>0 = only the k worst by static eval
    min_k_depth=2,      # only restrict when remaining depth >= this
    # nn weights
    w_empty=270.0, w_merges=700.0, w_mono=47.0, mono_pow=4.0, w_sum=11.0, sum_pow=3.5,
    # ov weights
    w_smooth=0.1, w_mono_ov=1.0, w_free=2.7, w_max=1.0,
)

LOST = -1e9
INF = float('inf')

_MOVE_L = [0] * 65536
_MOVE_R = [0] * 65536
_H = [0.0] * 65536      # nn heuristic per row
_SM = [0.0] * 65536     # ov smoothness per row
_M0 = [0.0] * 65536     # ov monotonicity: "decreasing" total per row
_M1 = [0.0] * 65536     # ov monotonicity: "increasing" total per row
_EM = [0] * 65536       # empty count per row
_MX = [0] * 65536       # max rank per row
_built = None


def _slide(line):
    tiles = [r for r in line if r]
    out, i = [], 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(min(tiles[i] + 1, 15)); i += 2
        else:
            out.append(tiles[i]); i += 1
    return out + [0] * (4 - len(out))


def configure(**kw):
    global _built
    CFG.update(kw)
    key = tuple(sorted(CFG.items()))
    if key == _built:
        return
    _built = key
    c = CFG
    pw_sum = [r ** c['sum_pow'] if r else 0.0 for r in range(16)]
    pw_mono = [r ** c['mono_pow'] for r in range(16)]
    for row in range(65536):
        line = [(row >> (4 * i)) & 0xF for i in range(4)]
        out = _slide(line)
        _MOVE_L[row] = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)
        o = _slide(line[::-1])[::-1]
        _MOVE_R[row] = o[0] | (o[1] << 4) | (o[2] << 8) | (o[3] << 12)

        empty = line.count(0)
        _EM[row] = empty
        _MX[row] = max(line)

        # ---- nn heuristic ----
        total = 0.0; merges = 0; prev = 0; counter = 0
        for rank in line:
            total += pw_sum[rank]
            if rank:
                if prev == rank:
                    counter += 1
                elif counter > 0:
                    merges += 1 + counter; counter = 0
                prev = rank
        if counter > 0:
            merges += 1 + counter
        ml = mr = 0.0
        for i in range(1, 4):
            if line[i - 1] > line[i]:
                ml += pw_mono[line[i - 1]] - pw_mono[line[i]]
            else:
                mr += pw_mono[line[i]] - pw_mono[line[i - 1]]
        _H[row] = (200000.0 + c['w_empty'] * empty + c['w_merges'] * merges
                   - c['w_mono'] * min(ml, mr) - c['w_sum'] * total)

        # ---- ovolve-style per-line terms ----
        nz = [r for r in line if r]
        _SM[row] = -float(sum(abs(nz[i] - nz[i + 1]) for i in range(len(nz) - 1)))
        t0 = t1 = 0.0
        cur, nxt = 0, 1
        while nxt < 4:
            while nxt < 4 and line[nxt] == 0:
                nxt += 1
            if nxt >= 4:
                nxt -= 1
            cv, nv = line[cur], line[nxt]
            if cv > nv:
                t0 += nv - cv
            elif nv > cv:
                t1 += cv - nv
            cur = nxt; nxt += 1
        _M0[row] = t0
        _M1[row] = t1


configure()


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


_MOVES = (_up, _down, _left, _right)


def _eval_nn(b):
    H = _H
    t = _transpose(b)
    return (H[b & 0xFFFF] + H[(b >> 16) & 0xFFFF] + H[(b >> 32) & 0xFFFF] + H[b >> 48]
            + H[t & 0xFFFF] + H[(t >> 16) & 0xFFFF] + H[(t >> 32) & 0xFFFF] + H[t >> 48])


def _eval_ov(b):
    t = _transpose(b)
    r0, r1, r2, r3 = b & 0xFFFF, (b >> 16) & 0xFFFF, (b >> 32) & 0xFFFF, b >> 48
    c0, c1, c2, c3 = t & 0xFFFF, (t >> 16) & 0xFFFF, (t >> 32) & 0xFFFF, t >> 48
    SM, M0, M1, EM, MX = _SM, _M0, _M1, _EM, _MX
    smooth = (SM[r0] + SM[r1] + SM[r2] + SM[r3] + SM[c0] + SM[c1] + SM[c2] + SM[c3])
    mono = (max(M0[r0] + M0[r1] + M0[r2] + M0[r3], M1[r0] + M1[r1] + M1[r2] + M1[r3])
            + max(M0[c0] + M0[c1] + M0[c2] + M0[c3], M1[c0] + M1[c1] + M1[c2] + M1[c3]))
    empty = EM[r0] + EM[r1] + EM[r2] + EM[r3]
    mx = max(MX[r0], MX[r1], MX[r2], MX[r3])
    c = CFG
    return (c['w_smooth'] * smooth + c['w_mono_ov'] * mono
            + c['w_free'] * (math.log(empty) if empty else -10.0) + c['w_max'] * mx)


def _pack(grid):
    b = 0; shift = 0
    for row in grid.map:
        for v in row:
            if v:
                b |= (v.bit_length() - 1) << shift
            shift += 4
    return b


class _Timeout(Exception):
    pass


class IntelligentAgent(BaseAI):
    def __init__(self):
        self.tt = {}
        self.nodes = 0
        self.deadline = 0.0
        self.ev = _eval_ov if CFG['heur'] == 'ov' else _eval_nn
        self.depth_reached = []

    def _max_value(self, b, depth, beta):
        if depth == 0:
            return self.ev(b)
        key = (b, depth)
        hit = self.tt.get(key)
        if hit is not None:
            return hit
        self.nodes += 1
        if self.nodes % 128 == 0 and time.process_time() > self.deadline:
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
            if len(self.tt) > 1500000:
                self.tt.clear()
            self.tt[key] = alpha
        return alpha

    def _chance_value(self, b, depth):
        cells = [s for s in range(0, 64, 4) if not (b >> s) & 0xF]
        if not cells:
            return self._max_value(b, depth - 1, INF)
        k = CFG['min_k']
        total = 0.0
        for tile, p in ((1, 0.9), (2, 0.1)):
            use = cells
            if k and len(cells) > k and depth >= CFG['min_k_depth']:
                ev = self.ev
                if time.process_time() > self.deadline:
                    raise _Timeout()
                use = sorted(cells, key=lambda s: ev(b | (tile << s)))[:k]
            beta = INF
            for s in use:
                v = self._max_value(b | (tile << s), depth - 1, beta)
                if v < beta:
                    beta = v
            total += p * beta
        return total

    def _search_root(self, b, depth, order):
        best_mv, best_val = None, LOST - 1
        for i in order:
            nb = _MOVES[i](b)
            if nb == b:
                continue
            v = self._chance_value(nb, depth)
            if v > best_val:
                best_val, best_mv = v, i
        return best_mv

    def getMove(self, grid):
        self.deadline = time.process_time() + CFG['budget']
        self.nodes = 0
        b = _pack(grid)
        legal = [i for i in range(4) if _MOVES[i](b) != b]
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]
        best, order, depth = legal[0], legal, 1
        try:
            while depth <= 64:
                mv = self._search_root(b, depth, order)
                if mv is not None:
                    best = mv
                    order = [mv] + [i for i in legal if i != mv]
                depth += 1
        except _Timeout:
            pass
        self.depth_reached.append(depth - 1)
        return best
