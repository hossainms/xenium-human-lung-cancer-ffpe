"""Shared helpers for the pipeline scripts.

Every script is a thin command-line wrapper around the ist_analysis package:

    python workflow/scripts/<step>.py --config config/config.yaml [--input ... --output ...]

Snakemake calls them with the input / output paths of each rule, but they also run by hand.
Settings come from the YAML config, never from constants inside the scripts. Printed paths are
shown relative to the home folder, so logs can be shared without exposing the local layout.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:   # the ist_analysis package, if it is not installed (pip install -e .)
    sys.path.insert(0, str(REPO_ROOT))
from ist_analysis import utils as xu  # noqa: E402
from ist_analysis.utils import mad_bounds  # noqa: E402,F401  (re-exported for the scripts)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Step:
    """Parsed arguments, config, and output helpers for one pipeline step."""

    def __init__(self, description: str, inputs: list[str], outputs: list[str]):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--config", default=str(REPO_ROOT / "config" / "config.yaml"))
        for name in inputs:
            parser.add_argument(f"--{name}", required=True, help=f"input: {name}")
        for name in outputs:
            parser.add_argument(f"--{name}", required=True, help=f"output: {name}")
        self.args = parser.parse_args()
        with open(self.args.config) as fh:
            self.config = yaml.safe_load(fh)
        self.results = Path(self.config["results"]).expanduser()
        self.figures = self.results / "figures"
        self.tables = self.results / "tables"
        for d in (self.figures, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.say(f"== {description}")

    # ---- paths -----------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        """An input or output path given on the command line."""
        p = Path(getattr(self.args, name.replace("-", "_"))).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_root(self) -> Path:
        return Path(self.config["data_root"]).expanduser()

    @property
    def sample_dir(self) -> Path:
        return self.data_root / self.config["sample"]

    @property
    def outs_dir(self) -> Path:
        return self.sample_dir / "outs"

    # ---- output ------------------------------------------------------------------------------
    @staticmethod
    def say(*parts) -> None:
        print(*parts, flush=True)

    def save_fig(self, fig, name: str) -> None:
        import matplotlib.pyplot as plt

        out = self.figures / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.say(f"  figure  {xu.display_path(out)}")

    def save_table(self, df, name: str, index: bool = True) -> None:
        out = self.tables / f"{name}.csv"
        df.to_csv(out, index=index)
        self.say(f"  table   {xu.display_path(out)}")

    def done(self, *outputs: Path) -> None:
        for p in outputs:
            self.say(f"  output  {xu.display_path(p)}")
        self.say(f"== done in {time.time() - self.t0:.0f} s")

