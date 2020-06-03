#!/bin/bash
set -e
for LANG in he hr bg fa bs fi
do
python3 make_job.py $LANG > en-$LANG.sh
#sbatch en-$LANG.sh
done
