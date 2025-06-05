#!/usr/bin/env bash
python3 ../vecalign/vecalign.py --alignment_max_size 10 --src $1 --tgt $2 --src_embed $3 $4 --tgt_embed $5 $6 | python3 ./align-cli.py -s $1 -t $2 > $7