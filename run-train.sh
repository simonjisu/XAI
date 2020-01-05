mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf mnist-resnet-roar \
    -dt mnist \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt resnet resnetcbam \
    -down \
    -bs 256 \
    -ns 10 \
    -cuda \
    -sd 73 > ./trainlog/mnist-resnet-roar.log &