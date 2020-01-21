mkdir -p trainlog
# plain roar
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-roar-eval \
#     -dt cifar10 \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-eval.log &

# rcd version: itu_r_bt2100
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-roar-rcd-eval \
#     -dt cifar10 \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -rcd itu_r_bt2100 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-rcd-eval.log &

# fill global mean: 
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cifar10-resnetsmall-roar-rcd-fgm-eval \
    -dt cifar10 \
    -et roar \
    -at random vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 30 \
    -cuda \
    -skip1 \
    -skip2 \
    -fgm \
    -rcd itu_r_bt2100 \
    -sd 73 > ./trainlog/cifar10-resnetsmall-roar-rcd-fgm-eval.log &

# kar version: do it after checked rcd is better
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-kar-eval \
#     -dt cifar10 \
#     -et kar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -skip2 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-kar-eval.log &