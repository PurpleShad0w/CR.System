#!/usr/bin/env bash
set -euo pipefail

ruff check --fix || true
ruff format
