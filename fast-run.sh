#!/bin/bash
mkdir -p trainlog
# 1: data-type
# 2: eval-type
# 3: experiments
# 4: is first train
if [ "$4" = true ]; then
    echo "This is first train"
    SKIP=""
else
    SKIP="-skip1 -skip2"
fi
# exp types
# 1: plain
# 2: rcd
# 3: rcd-fgm
# 4: rcd-noabs
if [ "$3" = "1" ]; then
    EXP=""
    EXP_STR="plain"
elif [ "$3" = "2" ]; then
    EXP="-rcd itu_r_bt2100"
    EXP_STR="rcd"
elif [ "$3" = "3" ]; then
    EXP="-rcd itu_r_bt2100 -fgm"
    EXP_STR="rcd-fgm"
elif [ "$3" = "rcd-noabs" ]; then
    EXP="-rcd itu_r_bt2100 -noabs"
    EXP_STR="rcd-noabs"
fi
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf "$1-$2-$EXP_STR" \
    -dt "$1" \
    -et "$2" \
    -at random vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 30 \
    -cuda \
    $SKIP \
    $EXP \
    -sd 73 > "./trainlog/$1-$2-$EXP_STR.log" &