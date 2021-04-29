import argparse

def argument_parsing(preparse=False):
    parser = argparse.ArgumentParser(description="Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path Settings
    parser.add_argument("-pp", "--prj_path", required=True,
                   help="Project Path: Insert your project dir path")
    parser.add_argument("-dp", "--data_path", required=True,
                   help="Data Path: Insert your project dir path")
    parser.add_argument("-rf", "--record_file", type=str, default="rc",
                help="Record directory Path: just insert the folder name,\
                    record all metrics to a table,\
                    it will automatically create under `prj_path`/trainlog/`record_file`.txt")
    # types
    parser.add_argument("-dt", "--data_type", type=str, default="mnist",
                   help="Dataset type: mnist, cifar10")
    parser.add_argument("-et", "--eval_type", type=str, default="roar",
                   help="Dataset type: roar, selectivity")
    parser.add_argument("-at", "--attr_type", nargs="+",
                   help="Attribution Method Type: random, deconv, gradcam, guidedgrad, relavance, vanillagrad, inputgrad, guided_gradcam. \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-at deconv gradcam`")
    parser.add_argument("-mt", "--model_type", nargs="+",
                   help="Model Type: cnn, resnet, resnetcbam, resnetanr \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-mt resnet resnetcbam resentanr`")
    # attribution details
    parser.add_argument("-fgm","--fill_global_mean", action="store_true",
                   help="Fill Global means to masked value if not will be replaced by zeros")
    parser.add_argument("-rcd", "--reduce_color_dim", type=str, default=None,
                   help="Reduce the color channel of dimension in the attribution maps by following methods.\
                        Methods: mean, rec601, itu_r_bt707, itu_r_bt2100")
    parser.add_argument("-noabs","--no_abs_grad", action="store_true",
                   help="when using gradient method, not to get absolute values of the attributions")

    # training
    parser.add_argument("-down","--download", action="store_true",
                   help="Whether to download the data")
    parser.add_argument("-bs","--batch_size", type=int, default=128,
                   help="Mini batch size")
    parser.add_argument("-ns","--n_step", type=int, default=10,
                   help="Total training step size")
    parser.add_argument("-cuda","--use_cuda", action="store_true",
                   help="Use Cuda")
    parser.add_argument("-sd","--seed", type=int, default=73,
                   help="Seed number")
    parser.add_argument("-skip1","--skip_first_train", action="store_true",
                   help="Skip first training")
    parser.add_argument("-skip2","--skip_second_masking", action="store_true",
                   help="Skip second masking")
    parser.add_argument("-skip3","--skip_third_eval", action="store_true",
                   help="Skip third evaluation")
    
    if preparse:
        return parser
    
    args = parser.parse_args()
    return args

def get_parser(data_type, option, no_attention=False, download=False):
    args_dict = {
        "mnist": {
            "roar": mnist_roar,
            "kar": mnist_kar
        },
        "cifar10": {
            "roar-plain": cifar10_roar_plain,  # no reduce color dimension
            "kar-rcd": cifar10_kar_rcd,
            "roar-rcd": cifar10_roar_rcd,
            "roar-rcd-fgm": cifar10_roar_rcd_fgm,
            "roar-rcd-noabs": cifar10_roar_rcd_noabs,
        }
    }
    parser = argument_parsing(preparse=True)
    parser.add_argument("-noatt", "--no_attention", action="store_true", help="No Attention Args")
    args = args_dict[data_type][option]
    if no_attention:
        args = args.replace("  resnetcbam  resnetanr", "")
        args += "\n-noatt"
    return parser.parse_args(args.replace("\n", "  ").split("  "))

mnist_roar= \
"""-pp  .
-dp  ./data
-rf  mnist-roar
-dt  mnist
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr"""

mnist_kar = \
"""-pp  .
-dp  ./data
-rf  mnist-kar
-dt  mnist
-et  kar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr"""

cifar10_roar_plain = \
"""-pp  .
-dp  ./data
-rf  cifar10-roar-plain
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr"""

cifar10_roar_rcd = \
"""-pp  .
-dp  ./data
-rf  cifar10-roar-rcd
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100"""

cifar10_kar_rcd = \
"""-pp  .
-dp  ./data
-rf  cifar10-kar-rcd
-dt  cifar10
-et  kar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100"""

cifar10_roar_rcd_fgm = \
"""-pp  .
-dp  ./data
-rf  cifar10-roar-rcd-fgm
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-fgm"""

cifar10_roar_rcd_noabs = \
"""-pp  .
-dp  ./data
-rf  cifar10-roar-rcd-noabs
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-noabs"""
