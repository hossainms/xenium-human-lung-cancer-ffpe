"""Notebook 03, part A1: download the Lung Cancer Atlas (LuCA core, CELLxGENE, 12.9 GB), resumable."""

import requests

from common import Step

step = Step("Notebook 03 A: download the LuCA core atlas", inputs=[], outputs=["output"])
url = step.config["reference"]["atlas_url"]
out = step.path("output")
total = int(requests.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
have = out.stat().st_size if out.exists() else 0
if have == total:
    step.say(f"  [skip] atlas complete ({total / 1e9:.1f} GB)")
else:
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "ab" if have and r.status_code == 206 else "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 << 20):
                fh.write(chunk)
    if out.stat().st_size != total:
        raise IOError("atlas download incomplete; re-run to resume")
step.done(out)
