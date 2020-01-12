mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cifar10-resnet-roar \
    -dt cifar10 \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 30 \
    -cuda \
    -sd 73 > ./trainlog/cifar10-resnet-roar.log &