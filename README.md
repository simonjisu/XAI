## XAI

* eXplainable AI
* AI college Recording Repo

## Requirements

```
python >= 3.6.8 (not 3.7.* yet)
pytorch >= 1.3.0
torchvision == 0.4.1
jupyter >= 1.0.0
voila >= 0.1.20
```

## Run in local

### 1. install requirements packages
* [[pytorch]](https://pytorch.org/) install with your environments
* [[jupyter]](https://jupyter.readthedocs.io/en/latest/install.html) better to install with anaconda
* [[voila]](https://voila.readthedocs.io/en/stable/install.html)
    * if you are using `jupyterlab` please run following commands
    ```
    jupyter labextension install @jupyter-voila/jupyterlab-preview
    ```

### 2. quick tutorial

clone or fork this project at first, please ensure make a `data` directory to download mnist, cifar10 datas.
```
$ cd [your directory]/XAI
$ mkdir ../data
```

and run following scripts to download all model weights(1.4GB), you can also download from [google drive](https://drive.google.com/file/d/1Av8B5gjKVL-vM-TvivKL1wNXmvaA4DMO/view?usp=sharing)

```
$ chmod 777 download-weight.sh
$ ./download-weight.sh
$ voila ./notebooks/Tutorial-01-XAI-Introduction.ipynb
```

after this go to browser http://localhost:8866

### 3. training from scratch

you can also train from scratch if you want. You can choose 3rd argument in "experiments" by following:
* 1: `plain` 
* 2: `rcd`
* 3: `rcd-fgm`
* 4: `rcd-noabs`

> options means:
>    * `plain`: basic setting
>    * `rcd`: gray scale for all attribution methods(means that recuding the color dimension to 1)
>    * `fgm`: fill the masks with global mean of all datas instead of zeros.
>    * `noabs`: not to absolute attribution scores in some methods

```bash
# 1: data-type
# 2: eval-type
# 3: experiments
# 4: if you train the model first time(for each data-type), 
#    ensure this variable to `true`
$ sh fast-run $1 $2 $3 $4
```

## About using `torchxai` Module

I'm updating "how to use".

## Project Process

What is going on this project? Please see `Project Process`!
* Project Process: [Project Page](https://github.com/simonjisu/XAI/projects/1)
