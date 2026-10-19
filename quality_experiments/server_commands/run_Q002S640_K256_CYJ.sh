#!/usr/bin/env bash
# Server-only Q002S640_K256 reconstruction. The caller supplies the immutable sparse prepared root for one CYJ case.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 CONDITION CROPPED_PREPARED_ROOT B6_MODEL APPROVED_DECODER NEW_OUTPUT" >&2; exit 2; }
CONDITION=$1; CROPPED_ROOT=$(realpath -m "$2"); B6_MODEL=$(realpath -m "$3"); SIGNAL=$(realpath -m "$4"); OUTPUT=$(realpath -m "$5")
CODE10W_ROOT=${CODE10W_ROOT:-/data/dengyz/code/Code_10w}
case "$CONDITION" in L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3) ;; *) echo "unsupported CYJ condition: $CONDITION" >&2; exit 2;; esac
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "$CODE10W_ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty Git worktree" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available()'
[[ -f "$B6_MODEL" && -f "$SIGNAL" ]] || { echo "Missing B6 model or approved decoder" >&2; exit 1; }
for stack in sax 2ch 4ch; do [[ -f "$CROPPED_ROOT/CYJ/$stack/observations.npz" || -f "$CROPPED_ROOT/$stack/observations.npz" ]] || { echo "Missing cropped observations for $stack" >&2; exit 1; }; done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing immutable output: $OUTPUT" >&2; exit 1; }
printf 'Q002S640_K256 CYJ reconstruction:\n  condition=%s\n  prepared=%s\n  output=%s\n' "$CONDITION" "$CROPPED_ROOT" "$OUTPUT"
python -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --cropped-prepared-root "$CROPPED_ROOT" --b6-model "$B6_MODEL" --signal-simulator "$SIGNAL" --output "$OUTPUT" --device cuda:0 --experiment-id Q002S640_K256 --anchor-batch-size 640
