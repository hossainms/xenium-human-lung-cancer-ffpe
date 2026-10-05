"""Step 1: download the 10x Xenium bundle (resumable) and extract it.

Complete files are skipped and partial ones resumed, so re-running is cheap. The marker output
(outs/experiment.xenium) is what downstream rules depend on.
"""

import shutil
import zipfile

import requests

from common import Step, xu

step = Step("Step 1: download and extract the Xenium bundle", inputs=[], outputs=["marker"])
cfg = step.config["download"]
sample = step.config["sample"]
data_dir = step.sample_dir
data_dir.mkdir(parents=True, exist_ok=True)
NO_GZIP = {"Accept-Encoding": "identity"}   # byte counts must match on-disk sizes

files = [f"{sample}_metrics_summary.csv", f"{sample}_gene_panel.json", f"{sample}_he_imagealignment.csv", f"{sample}_outs.zip"]
if cfg["he_image"]:
    files.append(f"{sample}_he_image.ome.tif")


def remote_size(url):
    r = requests.head(url, headers=NO_GZIP, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def download(url, dest):
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        step.say(f"  [skip] {dest.name} complete ({total / 1e9:.2f} GB)")
        return
    if have > total:
        dest.unlink()
        have = 0
    headers = {**NO_GZIP, **({"Range": f"bytes={have}-"} if have else {})}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if have and r.status_code != 206:   # server ignored the range request: start over
            have = 0
        with open(dest, "ab" if have else "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 << 20):
                fh.write(chunk)
    if dest.stat().st_size != total:
        raise IOError(f"{dest.name}: size mismatch; re-run to resume")
    step.say(f"  [done] {dest.name} ({total / 1e9:.2f} GB)")


for name in files:
    download(f"{cfg['base_url']}/{name}", data_dir / name)

zip_path = data_dir / f"{sample}_outs.zip"
outs = step.outs_dir
if not (outs / "experiment.xenium").exists():
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        tops = {m.filename.split("/")[0] for m in members}
        prefix = tops.pop() + "/" if len(tops) == 1 else ""   # strip a single top-level folder
        for m in members:
            rel = m.filename[len(prefix):]
            if not rel:
                continue
            target = outs / rel
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)
    step.say(f"  extracted {len(members):,} files to {xu.display_path(outs)}")
    if cfg["delete_zip"]:
        zip_path.unlink()
else:
    step.say("  [skip] bundle already extracted")

marker = step.path("marker")
marker.write_text("download complete\n")
step.done(marker)
