import argparse
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


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
    parser.add_argument("-at", "--attr_type", nargs="+", required=True,
                   help="Attribution Method Type: deconv, gradcam, guidedgrad, relavance, vanillagrad, inputgrad, guided_gradcam. \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-at deconv gradcam`")
    parser.add_argument("-mt", "--model_type", nargs="+", required=True,
                   help="Model Type: cnn, cnnwithcbam. \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-mt cnn cnnwithcbam`")
    # attribution details
    parser.add_argument("-rcd", "-reduce_color_dim", type=str,
                   help="Reduce the color channel of dimension in the attribution maps by following methods.\
                        Methods: rec601, itu-r_bt.707, itu0r_bt.2100")
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
    parser.add_argument("-vb","--verbose", type=int, default=0,
                   help="Verbose")
    
    if preparse:
        return parser
    
    args = parser.parse_args()
    return args


def build_img_dict(dataset):
    """
    build img dictionary by number 0~9
    imgs_dict = {
        number: {
            "imgs": tensor (number of img, 28, 28) 
            "index": LongTensor match to dataset
            }
        }
    """
    imgs_dict = {i:dict(imgs=[], index=[]) for i in range(10)}
    for i, (img, t) in enumerate(zip(dataset.data, dataset.targets)):
        imgs_dict[t.item()]["imgs"].append(img.float())
        imgs_dict[t.item()]["index"].append(i)

    for i in range(10):
        imgs_dict[i]["imgs"] = torch.stack(imgs_dict[i]["imgs"])
        imgs_dict[i]["index"] = torch.LongTensor(imgs_dict[i]["index"])
    return imgs_dict

def get_samples(imgs_dict, cat, sample_size=1, idx=None, return_idx=False):
    assert cat == "all" or isinstance(cat, int), "cat should be 'all' or int type"
    indices = []
    if cat == "all":
        temp = []
        for i in range(10):
            img_len = len(imgs_dict[i]["imgs"])
            idx = torch.randint(0, img_len, size=(sample_size, ))
            imgs = imgs_dict[i]["imgs"][idx].unsqueeze(1)
            temp.append(imgs)
            indices.append(imgs_dict[i]["index"][idx])
        samples = torch.cat(temp)
        labels = torch.arange(10)
    else:
        if idx is None:
            idx = torch.randint(0, len(imgs_dict[cat]["imgs"]), size=(sample_size, ))
        else:
            idx = torch.LongTensor([idx])
        indices.append(imgs_dict[i]["index"][idx])
        samples = imgs_dict[cat]["imgs"][idx].unsqueeze(1)
        labels = torch.LongTensor([cat])
    # Preprocessing
    samples = (samples - 0.5) / 0.5
    if return_idx:
        return samples, labels, indices
    else:
        return samples, labels

def model_predict(model, samples, get_switches=False):
    model.eval()
    outputs = model(samples, store=True)
    preds = outputs.argmax(1)
    if get_switches:
        convs_outputs, switches = model.forward_switches(samples, store=True)
        return preds, convs_outputs, switches
    else:
        return preds

def draw_numbers(preds, tensors, labels):
    assert tensors.size(0) == 10, "must be 10 numbers"
    fig, axes = plt.subplots(2, 5)

    for ax, pred, img, label in zip(axes.flatten(), preds, tensors, labels):
        title = f"target: {label.item()}\npredict: {pred.item()}"
        ax.imshow(img.squeeze()*0.5+0.5)
        ax.set_title(title)
        ax.axis("off")
    plt.show()

def draw_actmap(tensor, title, labels=None, vis_row=2, vis_channel=4, dpi=80, return_fig=False, **kwargs):
    """
    vis_row: related with batch
    vis_channel: related with channel
    """
    if kwargs["spaces"] is None:
        (o_wspace, o_hspace), (i_wspace, i_hspace) = [(0.1, 0.2), (0.05, 0.05)]
    else:
        (o_wspace, o_hspace), (i_wspace, i_hspace) = kwargs["spaces"]
    (fig_h, fig_w) = (32, 16) if kwargs["figsize"] is None else kwargs["figsize"]
    i_title_size = 10 if kwargs["i_title_size"] is None else kwargs["i_title_size"]
    
    tensor = tensor.detach()
    B, C, H, W = tensor.size()
    if C == 1:
        a, b = (1, 1)
    else:
        a, b = (vis_channel, C//vis_channel)
    
    fig = plt.figure(figsize=(fig_h, fig_w), dpi=dpi)
    fig.suptitle(title, fontsize=20)
    o_grid = gridspec.GridSpec(vis_row, B//vis_row, wspace=o_wspace, hspace=o_hspace)
    for k, o_g in enumerate(o_grid):
        ax = fig.add_subplot(o_g)
        if labels is not None:
            t = f"target={labels[k]}"
            ax.set_title(t, fontsize=i_title_size)
        ax.axis("off")

        i_grid = gridspec.GridSpecFromSubplotSpec(a, b,
                subplot_spec=o_g, wspace=i_wspace, hspace=i_hspace)

        axes = []
        for i, img in enumerate(tensor[k]):
            ax = fig.add_subplot(i_grid[i])
            im = ax.imshow(img, cmap="coolwarm")
            ax.axis("off")
            axes += [ax]
        fig.colorbar(im, ax=axes, orientation='vertical', pad=0.05)
    plt.show()
    if return_fig:
        return fig

def get_max_activation(act_map, max_num):
    B, C, H, W = act_map.size() 
    sort_indices = torch.abs(act_map).sum(-1).sum(-1).sort(-1)[1]
    return act_map[sort_indices < max_num].view(B, -1, H, W)

def draw_act_max(layer_name, model, max_num, spaces, figsize, vis_row, vis_channel, dpi, i_title_size=10, labels=None):
    act_map = model.activation_maps[layer_name]
    if max_num == -1:
        max_num = act_map.size(1)
    act_map_max = get_max_activation(act_map, max_num=max_num)
    draw_actmap(act_map_max, layer_name+f": max {max_num}", 
                labels=labels, vis_row=vis_row, 
                vis_channel=vis_channel, dpi=dpi, 
                spaces=spaces, i_title_size=i_title_size, figsize=figsize)

def draw_attribution(tensor, title, labels=None, vis_row=2, dpi=80):
    tensor = tensor.detach()
    B, C, H, W = tensor.size()
    fig, axes = plt.subplots(vis_row, B//vis_row, figsize=((B//vis_row)*4, vis_row*4), dpi=dpi)
    fig.suptitle(title, fontsize=20, y=1.01)
    if B == 1:
        axes = [axes]
    for k, ax in enumerate(axes.flatten()):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(tensor[k].squeeze(), cmap="coolwarm")
        fig.colorbar(im, cax=cax, orientation='vertical')
        if labels is not None:
            t = f"target={labels[k]}"
        else:
            t = title
        ax.set_title(t, fontsize=16)
    plt.tight_layout()
    plt.show()