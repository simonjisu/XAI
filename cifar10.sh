mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cifar10-resnetsmall-roar \
    -dt cifar10 \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr\
    -down \
    -rcd mean \
    -bs 256 \
    -ns 35 \
    -cuda \
    -sd 73 > ./trainlog/cifar10-resnetsmall-roar.log &