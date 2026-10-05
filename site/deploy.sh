#!/usr/bin/env bash
# Publish the built site (site/build_site.py) to the gh-pages branch, which GitHub Pages serves.
# The branch holds one commit, replaced on every deploy, so old viewer data does not pile up in history.
#
#   bash site/deploy.sh [site folder, default ~/data/xenium_lung/site]     # from the repository root
set -euo pipefail

SITE="${1:-$HOME/data/xenium_lung/site}"
REMOTE="$(git remote get-url origin)"
[ -f "$SITE/index.html" ] || { echo "No built site in $SITE: run site/build_site.py first"; exit 1; }

cd "$SITE"
rm -rf .git
git init -q -b gh-pages
git add -A
git commit -q -m "Publish the project website" -m "Author: MD Shakhaowat Hossain"
git push -q -f "$REMOTE" gh-pages
rm -rf .git
echo "Published to gh-pages: https://hossainms.github.io/xenium-human-lung-cancer-ffpe/"
