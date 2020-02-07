#!/bin/bash
# https://drive.google.com/file/d/1Av8B5gjKVL-vM-TvivKL1wNXmvaA4DMO/view?usp=sharing
python ./src/downutils.py -id "1Av8B5gjKVL-vM-TvivKL1wNXmvaA4DMO" -to "./weights.tar"
mkdir -p trained
tar -xvf ./weights.tar -C trained
rm ./weight.tar