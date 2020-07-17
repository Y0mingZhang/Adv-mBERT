#!/bin/bash
set -e
for FILE in dual_std0.1_3epochs_skipner.py dual_std0.5_3epochs_skipner.py dual_td0.1_3_skipner_nt.py
do
for LANG in he bg fa fi
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
#sbatch en-$LANG.sh
done
done
