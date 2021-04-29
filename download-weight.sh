#!/bin/bash
# https://drive.google.com/file/d/1Av8B5gjKVL-vM-TvivKL1wNXmvaA4DMO/view?usp=sharing

trained_dir_name="trained"
data_dir_name="data"

if [ ! -d $trained_dir_name ]; then
  mkdir ${trained_dir_name}
fi

if [ ! -d $data_dir_name ]; then
  mkdir ${data_dir_name}
fi

python ./src/downutils.py -id "1Av8B5gjKVL-vM-TvivKL1wNXmvaA4DMO" -to "./weights.tar"
tar -xvf ./weights.tar -C trained
rm ./weight.tar