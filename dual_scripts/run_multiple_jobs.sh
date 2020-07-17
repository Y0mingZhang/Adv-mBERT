#!/bin/bash
set -e
for FILE in dual_td0_3_ner.py dual_td0_3_skipner.py dual_td0.1_3_skipner.py
do
for LANG in he bg fa fi
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
#sbatch en-$LANG.sh
done
done
