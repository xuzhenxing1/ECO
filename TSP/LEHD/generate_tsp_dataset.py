import argparse
import os
from typing import List

import numpy as np


def nearest_neighbor_tour(coords: np.ndarray, start: int = 0) -> List[int]:
    """Return a tour as a list of node indices (0-based) using nearest-neighbor heuristic."""
    n = coords.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [int(start)]
    visited[start] = True

    for _ in range(n - 1):
        last = tour[-1]
        diff = coords - coords[last]
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
        d2[visited] = np.inf
        nxt = int(np.argmin(d2))
        tour.append(nxt)
        visited[nxt] = True

    return tour


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LEHD-style TSP dataset text file.")
    parser.add_argument("--problem_size", type=int, default=500)
    parser.add_argument("--num_instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dataset path. Default: TSP/data/gen_TSP{N}_n{K}.txt",
    )
    parser.add_argument("--coord_min", type=float, default=0.0)
    parser.add_argument("--coord_max", type=float, default=1.0)
    parser.add_argument("--float_precision", type=int, default=6)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    if args.out is None:
        args.out = os.path.join(data_dir, f"gen_TSP{args.problem_size}_n{args.num_instances}.txt")

    rng = np.random.default_rng(args.seed)
    n = args.problem_size
    k = args.num_instances
    fmt = f"{{:.{args.float_precision}f}}"

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(k):
            coords = rng.uniform(args.coord_min, args.coord_max, size=(n, 2)).astype(np.float64)
            start = int(rng.integers(0, n))
            tour = nearest_neighbor_tour(coords, start=start)

            # Format: x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1
            tokens: List[str] = []
            for x, y in coords:
                tokens.append(fmt.format(float(x)))
                tokens.append(fmt.format(float(y)))

            tokens.append("output")
            tokens.extend(str(node + 1) for node in tour)
            tokens.append(str(tour[0] + 1))

            f.write(" ".join(tokens) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == k:
                print(f"generated {i + 1}/{k} instances -> {args.out}")


if __name__ == "__main__":
    main()
