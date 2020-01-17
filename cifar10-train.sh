mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -dt cifar10 \
    -mt resnet resnetcbam resnetanr \
    -down \
    -rcd mean \
    -bs 128 \
    -ns 40 \
    -cuda \
    -skip2 \
    -sd 73 > ./trainlog/cifar10-train.log &