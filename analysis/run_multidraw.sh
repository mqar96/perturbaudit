#!/bin/sh
SRC=$1; TAG=$2; P=$3; shift 3
for s in 11 12 13 14 15; do
  D=derived/panels/${TAG}_d$s
  if [ ! -f $D/panel.npz ]; then PANEL_SEED=$s CELL_FRAC=0.8 python3 perturbaudit/scripts/build_panel_k562.py $D $P 20 20 > /dev/null; fi
  for m in "$@"; do
    [ -f $D/W_$m.npy ] || python3 perturbaudit/scripts/run_method.py $m $D
  done
  python3 perturbaudit/scripts/audit_panel.py $D > /dev/null
  echo "draw $s done"
done
