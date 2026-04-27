#!/usr/bin/env bash
# Frontend quality checks: formatting verification

set -e

FRONTEND_DIR="$(dirname "$0")/frontend"

cd "$FRONTEND_DIR"

if ! command -v node &>/dev/null; then
  echo "Error: node not found. Install Node.js to run frontend checks." >&2
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo "Checking frontend formatting with Prettier..."
npx prettier --check .

echo "All frontend checks passed."
