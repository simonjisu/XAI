mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf mnist-resnetsmall-roar \
    -dt mnist \
    -et roar \
    -at random vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 10 \
    -cuda \
    -sd 73 > ./trainlog/mnist-resnetsmall-roar.log &