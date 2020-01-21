# XAI

* eXplainable AI
* AI college Recording Repo

# Requirements

```
python >= 3.6.8 (not 3.7.* yet)
pytorch >= 1.3.0
torchvision == 0.4.1
jupyter >= 1.0.0
voila >= 0.1.20
```

## run in local

### 1. install requirements packages
* [[pytorch]](https://pytorch.org/) install with your environments
* [[jupyter]](https://jupyter.readthedocs.io/en/latest/install.html) better to install with anaconda
* [[voila]](https://voila.readthedocs.io/en/stable/install.html)
    * if you are using `jupyterlab` please run following commands
    ```
    jupyter labextension install @jupyter-voila/jupyterlab-preview
    ```

### 2. quick tutorial

clone or fork this project at first, please ensure make a data directory to download mnist, cifar10 datas

```
$ cd [your directory]/XAI
$ mkdir ../data
$ voila Tutorial-01-XAI-Introduction.ipynb
```

after this go to browser http://localhost:8866
please visit http://soopace.com:13845 to see what's going on(open for 01.21~01.30)

* avaliable contents: Tutorial-02

### 3. training from scratch

## Project Process

What is going on this project? Please see `Project Process`!
* Project Process: [Project Page](https://github.com/simonjisu/XAI/projects/1)
* How to Use(Building)

#!/bin/bash
if [ "$1" = "run" ] && [ "$2" = "notebook" ]
then
        echo "jupyter notebook start"
        nohup jupyter notebook &
elif [ "$1" = "run" ] && [ "$2" = "lab" ]
then
        echo "jupyter lab start"
        nohup jupyter lab &
elif [ "$1" = "stop" ] && [ "$2" = "notebook" ]
then
	pkill -9 -ef jupyter-notebook
	echo "jupyter notebook stop"
elif [ "$1" = "stop" ] && [ "$2" = "lab" ]
then
	pkill -9 -ef jupyter-lab
	echo "jupyter lab stop"
else
        echo "insert args: 1) run / stop 2) notebook / lab"
fi
