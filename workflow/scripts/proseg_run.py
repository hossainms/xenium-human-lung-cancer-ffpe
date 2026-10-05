"""Notebook 02, part B: re-segment the cells with Proseg (~30 min, all cores, ~13 GB RAM).

Proseg's Xenium preset reads transcripts.parquet directly (gene transcripts, qv >= 20) and starts from the
10x nuclei. If config proseg.reuse points to an earlier run's proseg-output.zarr, that run is linked
instead of repeating it (the run is deterministic for a fixed input, so this only saves time).
"""

import subprocess
from pathlib import Path

from common import Step, xu

step = Step("Notebook 02 B: Proseg re-segmentation", inputs=["marker"], outputs=["output"])
cfg = step.config["proseg"]
out = Path(step.args.output).expanduser()
out.parent.mkdir(parents=True, exist_ok=True)

reuse = Path(cfg["reuse"]).expanduser() if cfg.get("reuse") else None
if reuse and reuse.exists():
    if out.is_symlink() or out.exists():
        out.unlink()
    out.symlink_to(reuse, target_is_directory=True)
    step.say(f"  reusing the earlier Proseg run at {xu.display_path(reuse)} (set proseg.reuse: null to re-run)")
else:
    binary = Path(cfg["binary"]).expanduser()
    args = [str(binary), "--xenium", str(step.outs_dir / "transcripts.parquet"), "--nthreads", str(cfg["threads"]),
            "--output-counts", "counts.csv.gz", "--output-cell-metadata", "cell-metadata.csv.gz",
            "--output-cell-polygons", "cell-polygons.geojson.gz", "--exclude-spatialdata-transcripts",
            "--output-spatialdata", out.name, "--overwrite"]
    step.say("  running: proseg " + " ".join(xu.display_path(Path(a)) if a.startswith("/") else a for a in args[1:]))
    with open(out.parent / "proseg_run.log", "w") as log:   # Proseg's own log; it may contain local paths
        subprocess.run(args, cwd=out.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
step.done(out)
