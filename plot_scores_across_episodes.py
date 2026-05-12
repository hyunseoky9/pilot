import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SEED_RE = re.compile(r"seed:\s*(\d+)")
EPISODE_SCORE_RE = re.compile(
    r"Episode\s+(\d+),\s+Learning Rate:.*?Avg Performance:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_training_output(log_path: Path) -> dict[int, list[tuple[int, float]]]:
    """Parse the training log and return episode-performance pairs grouped by seed."""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    current_seed: int | None = None

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            seed_match = SEED_RE.search(line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                continue

            score_match = EPISODE_SCORE_RE.search(line)
            if score_match and current_seed is not None:
                episode = int(score_match.group(1))
                score = float(score_match.group(2))
                by_seed[current_seed].append((episode, score))

    return by_seed


def plot_seed_curves(
    seed_data: dict[int, list[tuple[int, float]]],
    title: str = "Performance Across Episodes by Seed",
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    if not seed_data:
        raise ValueError("No seed/episode performance data found in the log file.")

    plt.figure(figsize=(13, 7))

    for seed in sorted(seed_data):
        pairs = sorted(seed_data[seed], key=lambda x: x[0])
        episodes = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        plt.plot(episodes, scores, marker="o", markersize=3, linewidth=1.5, label=f"seed {seed}")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Performance Score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Avg Performance vs Episode for all seeds from a PPO training log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="trainingoutput.txt",
        help="Path to training output text file (default: trainingoutput.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results/performance_by_seed.png",
        help="Path for output image (default: results/performance_by_seed.png)",
    )
    parser.add_argument(
        "--title",
        default="Performance Across Episodes by Seed",
        help="Plot title",
    )

    args = parser.parse_args()
    log_path = Path(args.log_file)
    output_path = Path(args.output)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    seed_data = parse_training_output(log_path)
    plot_seed_curves(seed_data, title=args.title, output_path=output_path, show=False)

    total_points = sum(len(v) for v in seed_data.values())
    print(f"Parsed {len(seed_data)} seeds and {total_points} episode-score points.")
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
