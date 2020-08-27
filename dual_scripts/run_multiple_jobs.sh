#!/bin/bash
set -e
for FILE in dual_std0.1_lowlr.py
do
for LANG in he bg fa fi af ar bn de el es et fr hi hu id it ms nl pt ru ta tl tr vi
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
sbatch $FILE-en-$LANG.sh
done
done
