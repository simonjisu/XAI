mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cnn-roar \
    -dt mnist \
    -et roar \
    -at vanillagrad inputgrad gradcam \
    -mt resnet resnetcbam \
    -down \
    -bs 256 \
    -ns 10 \
    -cuda \
    -sd 73 > ./trainlog/cnn.log &
