mkdir -p trainlog
# plain roar
# nohup python -u main.py \
#     -pp $HOME/code/XAI \
#     -dp $HOME/code/data \
#     -rf mnist-resnetsmall-roar-eval \
#     -dt mnist \
#     -et roar \
#     -at random vanillagrad inputgrad guidedgrad gradcam \
#     -mt resnet resnetcbam resnetanr \
#     -down \
#     -bs 256 \
#     -ns 10 \
#     -cuda \
#     -skip1 \
#     -sd 73 > ./trainlog/mnist-resnetsmall-roar-eval.log &

# plain kar
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf mnist-resnetsmall-kar-eval \
    -dt mnist \
    -et kar \
    -at random vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 10 \
    -cuda \
    -skip1 \
    -skip2 \
    -sd 73 > ./trainlog/mnist-resnetsmall-kar-eval.log &