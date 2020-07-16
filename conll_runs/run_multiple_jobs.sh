#!/bin/bash
set -e
for FILE in *.py
do
for LANG in es nl de
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
#sbatch en-$LANG.sh
done
done
