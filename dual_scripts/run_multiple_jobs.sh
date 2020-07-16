#!/bin/bash
set -e
for FILE in dual*.py
do
for LANG in he bg fa fi
do
python3 $FILE $LANG > $FILE-en-$LANG.sh
#sbatch en-$LANG.sh
done
done
