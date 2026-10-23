# Benchmark harness for exp_agent variants (not part of the submission).
# Usage: python3 bench.py <games> <max_moves> '<json cfg>' ['<json cfg>' ...]
# Plays games with the same rules as GameManager (random computer, 0.9/0.1 tiles,
# CPU-time limit of 0.25 per turn pair), capped at max_moves player moves.
# Prints every game as it finishes.

import json
import math
import multiprocessing as mp
import random
import sys
import time


def play(args):
    cfg, seed, max_moves = args
    import importlib
    from Grid import Grid
    cfg = dict(cfg)
    exp_agent = importlib.import_module(cfg.pop('module', 'exp_agent'))
    if hasattr(exp_agent, 'configure'):
        exp_agent.configure(**cfg)
    random.seed(seed)
    agent = exp_agent.IntelligentAgent()
    g = Grid(4)
    for _ in range(2):
        g.setCellValue(random.choice(g.getAvailableCells()), 2 if random.random() < 0.9 else 4)
    moves = 0
    worst = 0.0
    status = 'cap'
    while moves < max_moves:
        if not g.canMove():
            status = 'dead'
            break
        t = time.process_time()
        mv = agent.getMove(g.clone())
        dt = time.process_time() - t
        worst = max(worst, dt)
        if dt > 0.25:
            status = 'TIMEOUT'
            break
        if mv is None or not g.move(mv):
            status = 'INVALID'
            break
        moves += 1
        cells = g.getAvailableCells()
        if cells:
            g.setCellValue(random.choice(cells), 2 if random.random() < 0.9 else 4)
    d = getattr(agent, 'depth_reached', [])
    return (json.dumps(cfg, sort_keys=True), seed, g.getMaxTile(), status, moves, worst,
            sum(d) / len(d) if d else 0)


if __name__ == '__main__':
    n = int(sys.argv[1]); cap = int(sys.argv[2])
    cfgs = [json.loads(a) for a in sys.argv[3:]] or [{}]
    jobs = [(c, s, cap) for c in cfgs for s in range(n)]
    results = {}
    with mp.Pool(min(len(jobs), 10)) as pool:
        for r in pool.imap_unordered(play, jobs):
            print("%s seed=%d max=%d %s moves=%d worst=%.3fs avgdepth=%.1f" % r, flush=True)
            results.setdefault(r[0], []).append(r[2])
    print()
    for k, tiles in results.items():
        top = sorted(tiles)[-5:]
        print(k, sorted(tiles), "mean log2 = %.2f" % (sum(math.log2(t) for t in tiles) / len(tiles)))
