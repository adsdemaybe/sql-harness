#!/usr/bin/env bash
# Fetch the AdventureWorks OLTP sample as SQLite into data/adventureworks.db.
# Source: https://github.com/demo-db/adventureworks (MIT, Microsoft sample, converted;
# keeps all 71 tables plus the original primary and foreign keys).
set -euo pipefail

BASE=https://raw.githubusercontent.com/demo-db/adventureworks/main/artifacts
SHA256=6a105e1982becfe003fc7a307d7cad9d738390cedd79d166c2167fd817168109
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/adventureworks.db"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

for part in 0001 0002; do
  curl -fsSL -o "$tmp/part-$part" "$BASE/adventureworks.sqlite.gz.part-$part"
done
cat "$tmp/part-0001" "$tmp/part-0002" | gunzip -c > "$tmp/adventureworks.db"

actual=$(shasum -a 256 "$tmp/adventureworks.db" | cut -d' ' -f1)
if [ "$actual" != "$SHA256" ]; then
  echo "checksum mismatch: got $actual, expected $SHA256" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
mv "$tmp/adventureworks.db" "$DEST"
echo "wrote $DEST"
