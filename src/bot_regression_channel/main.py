from dataclasses import dataclass
from typing import Iterable

@dataclass
class Stats:
    """Simple stats container."""
    count: int
    mean: float

def mean_of(xs: Iterable[float]) -> Stats:
    total = 0.0
    n = 0
    for x in xs:
        total += float(x)
        n += 1
    if n == 0:
        raise ValueError("mean_of() received empty iterable")
    return Stats(count=n, mean=total / n)

def run_example() -> None:
    s = mean_of([1, 2, 3, 4])
    print(f"n={s.count} mean={s.mean:.2f}")

if __name__ == "__main__":
    run_example()
