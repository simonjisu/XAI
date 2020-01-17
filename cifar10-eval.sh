mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cifar10-eval-roar-continue \
    -dt cifar10 \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt resnetanr \
    -down \
    -rcd mean \
    -bs 128 \
    -ns 35 \
    -cuda \
    -skip1 \
    -sd 73 > ./trainlog/cifar10-eval-continue.log &

# no color reduce version
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnet-roar-nocolor \
#     -dt cifar10 \
#     -et roar \
#     -at vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 128 \
#     -ns 35 \
#     -cuda \
#     -skip1 \
#     -sd 73 > ./trainlog/cifar10-resnet-roar-nocolor.log &