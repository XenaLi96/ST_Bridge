#!/usr/bin/env bash
set -euo pipefail

# Run after installing the package. Each invocation uses a fresh temp directory.
bridge_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bridge_demo_dir=$(mktemp -d)
trap 'rm -rf "$bridge_demo_dir"' EXIT
bridge-st --version
bridge-st make-demo --out "$bridge_demo_dir/data"
bridge-st validate --data "$bridge_demo_dir/data/synthetic.npz" --targets
bridge-st train --data "$bridge_demo_dir/data/synthetic.npz" --config "$bridge_source_dir/configs/demo.json" --out "$bridge_demo_dir/run" --device cpu
bridge-st predict --data "$bridge_demo_dir/data/synthetic_inputs.npz" --checkpoint "$bridge_demo_dir/run/model.pt" --out "$bridge_demo_dir/predictions.npz" --device cpu
bridge-st evaluate --reference "$bridge_demo_dir/data/synthetic.npz" --predictions "$bridge_demo_dir/predictions.npz" --out "$bridge_demo_dir/metrics" --split test
cat "$bridge_demo_dir/metrics/summary.json"
