#!/bin/bash
set -e
for FILE in make_job*.py
do
for LANG in he bg fa fi
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
#sbatch en-$LANG.sh
done
done
