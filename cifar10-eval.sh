mkdir -p trainlog
## absolute gradient in *grad methods
# plain version
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

# rcd version
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
#     -skip2 \
#     -rcd itu_r_bt2100 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-rcd-eval.log &

# rcd with fgm
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-roar-rcd-fgm-eval \
#     -dt cifar10 \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -skip2 \
#     -fgm \
#     -rcd itu_r_bt2100 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-rcd-fgm-eval.log &

# kar version: 
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-kar-rcd-eval \
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
#     -rcd itu_r_bt2100 \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-kar-rcd-eval.log &

## ---
## not absoulute gradient
# rcd version: itu_r_bt2100
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cifar10-resnetsmall-roar-noabs-rcd-eval \
    -dt cifar10 \
    -et roar \
    -at random vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 30 \
    -cuda \
    -skip1 \
    -rcd itu_r_bt2100 \
    -noabs \
    -sd 73 > ./trainlog/cifar10-resnetsmall-roar-noabs-rcd-eval.log &

# plain roar 
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-roar-noabs-eval \
#     -dt cifar10 \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -skip2 \
#     -noabs \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-noabs-eval.log &

# fill global mean: 
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf cifar10-resnetsmall-roar-noabs-rcd-fgm-eval \
#     -dt cifar10 \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 30 \
#     -cuda \
#     -skip1 \
#     -skip2 \
#     -fgm \
#     -rcd itu_r_bt2100 \
#     -noabs \
#     -sd 73 > ./trainlog/cifar10-resnetsmall-roar-noabs-rcd-fgm-eval.log &
