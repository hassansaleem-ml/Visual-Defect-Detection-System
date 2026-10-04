#!/usr/bin/env bash
# One command, end to end. Needs data/raw/{train,test}/ (KolektorSDD2) and `pip install -r requirements.txt`.
set -euo pipefail
python -m src.prepare_data --raw data/raw              # audit + split
python -m src.train --raw data/raw                     # transfer learning -> models/best_model.pth
python -m src.evaluate --raw data/raw --split val      # dev sanity check (reports/val, examples/val)
python -m src.evaluate --raw data/raw                  # FINAL test evaluation - run ONCE
python -m src.fill_results                             # README/report tables from the generated files
python -m pytest -q
echo "Done. Now inspect examples/false_negatives and examples/false_positives (see report sections 14-16)."
