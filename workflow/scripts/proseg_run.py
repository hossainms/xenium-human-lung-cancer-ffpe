"""Notebook 02, part B: re-segment the cells with Proseg (~30 min, all cores, ~13 GB RAM) (ist_analysis.proseg).

If config proseg.reuse points to an earlier run's proseg-output.zarr, that run is linked instead of
repeating it (the run is deterministic for a fixed input, so this only saves time).
"""

from pathlib import Path

from common import Step, xu
from ist_analysis import proseg

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
    step.say(f"  running Proseg on {xu.display_path(step.outs_dir / 'transcripts.parquet')} ({cfg['threads']} threads)")
    proseg.run_proseg(Path(cfg["binary"]).expanduser(), step.outs_dir / "transcripts.parquet", out.parent, threads=cfg["threads"],
                      log_name="proseg_run.log")   # Proseg's own log; it may contain local paths
step.done(out)
