mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf mnist-resnet-roar \
    -dt mnist \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam resnetanr \
    -down \
    -bs 256 \
    -ns 10 \
    -cuda \
    -skip1 \
    -sd 73 > ./trainlog/mnist-resnet-roar-eval.log &