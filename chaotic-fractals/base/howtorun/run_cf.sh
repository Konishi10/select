#!/bin/bash

set -e

target_dir="cf_t"

python scripts/chaotic_fractals.py \
    --dir "$target_dir" \
    --config "test_suite_ours" \
    --trainer_type "cf" \
    --gt_name "fdb_0"
