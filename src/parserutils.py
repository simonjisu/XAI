from .utils import argument_parsing

def get_parser(data_type, option):
    args_dict = {
        "mnist": {
            "roar": mnist_roar,
            "kar": mnist_kar
        },
        "cifar10": {
            "roar": cifar10_roar,  # no reduce color dimension
            "kar-rcd": cifar10_kar_rcd,
            "roar-rcd": cifar10_roar_rcd,
            "roar-rcd-fgm": cifar10_roar_rcd_fgm,
            "roar-noabs": cifar10_roar_noabs,
            "roar-noabs-rcd": cifar10_roar_noabs_rcd,
            "roar-noabs-rcd-fgm": cifar10_roar_noabs_rcd_fgm
            
        }
    }
    parser = argument_parsing(preparse=True)
    return parser.parse_args(args_dict[data_type][option])

mnist_roar= \
"""-pp  ../XAI
-dp  ../data
-rf  mnist-resnetsmall-roar-eval
-dt  mnist
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-down""".replace("\n", "  ").split("  ")

mnist_kar = \
"""-pp  ../XAI
-dp  ../data
-rf  mnist-resnetsmall-kar-eval
-dt  mnist
-et  kar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-down""".replace("\n", "  ").split("  ")

cifar10_roar = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-down""".replace("\n", "  ").split("  ")

cifar10_roar_rcd = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-rcd-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-down""".replace("\n", "  ").split("  ")

cifar10_kar_rcd = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-kar-rcd-eval
-dt  cifar10
-et  kar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-down""".replace("\n", "  ").split("  ")

cifar10_roar_rcd_fgm = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-rcd-fgm-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-fgm
-down""".replace("\n", "  ").split("  ")

cifar10_roar_noabs = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-noabs-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-noabs
-down""".replace("\n", "  ").split("  ")

cifar10_roar_noabs_rcd = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-noabs-rcd-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-noabs
-down""".replace("\n", "  ").split("  ")

cifar10_roar_noabs_rcd_fgm = \
"""-pp  ../XAI
-dp  ../data
-rf  cifar10-resnetsmall-roar-noabs-rcd-fgm-eval
-dt  cifar10
-et  roar
-at  random  vanillagrad  inputgrad  guidedgrad  gradcam
-mt  resnet  resnetcbam  resnetanr
-rcd  itu_r_bt2100
-noabs
-fgm
-down""".replace("\n", "  ").split("  ")
