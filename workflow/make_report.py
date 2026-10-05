"""Build the Snakemake HTML report with local paths removed, so it can be shared.

Snakemake writes the absolute path of every output into the report. This wrapper builds it, then
replaces the home folder with "~" (plain and JSON-escaped forms) and checks that none is left.

    python workflow/make_report.py [report.html]       # run from the repository root, in snakemake_env
"""

import os
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "report.html")
tmp = out.with_name(out.stem + ".tmp.html")
subprocess.run(["snakemake", "--report", str(tmp)], check=True)

home = os.path.expanduser("~")
html = tmp.read_text(encoding="utf-8")
for form in (home, home.replace("/", "\\/")):
    html = html.replace(form, "~")
if home in html:
    raise SystemExit("local paths still present; report not written")
out.write_text(html, encoding="utf-8")
tmp.unlink()
print(f"Report written to {out.name} (local paths replaced with ~)")
