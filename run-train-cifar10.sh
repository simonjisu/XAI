mkdir -p trainlog
nohup python -u main.py \
    -pp $HOME/code/XAI \
    -dp $HOME/code/data \
    -rf cnn-roar-cifar10 \
    -dt cifar10 \
    -et roar \
    -at vanillagrad inputgrad guidedgrad gradcam \
    -mt cnnwithcbam \
    -down \
    -bs 256 \
    -ns 15 \
    -cuda \
    -sd 73 > ./trainlog/cnnwithcbam.log &
