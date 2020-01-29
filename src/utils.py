__author__ = "simonjisu"

from .trainsettings import ModelTranier
import argparse
import ipywidgets as widgets
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


class Explorer(ModelTranier):
    def __init__(self, args):
        super(Explorer, self).__init__()
        # list
        self.model_type = args.model_type
        self.attr_type = args.attr_type
        # string
        self.data_type = args.data_type
        self.eval_type = args.eval_type
        # bool
        self.reduce_color_dim = args.reduce_color_dim
        self.fill_global_mean = args.fill_global_mean
        self.no_abs_grad = args.no_abs_grad
        # path settings
        self.prj_path = Path(args.prj_path)
        self.sv_main_path = self.prj_path/"trained"/self.data_type
        self.record_main_path = self.prj_path/"trainlog"
        if self.reduce_color_dim is not None:
            self.sv_attr_path = self.sv_main_path/self.eval_type/self.reduce_color_dim
        else:
            self.sv_attr_path = self.sv_main_path/self.eval_type/"plain"
        if self.no_abs_grad:
            self.sv_attr_path = self.sv_attr_path/"noabs"
        self.record_path = self.record_main_path/f"{args.record_file}.txt" 
        
        _, self.test_dataset, *_ = self.build_dataset(args)
        self.img_dict, self.idx_to_class = self.build_img_dict(self.test_dataset)
        # attenion options
        self.attention_dict_default = {"none": None}
        self.attention_dict = self.attention_dict_default
        self.attn_option = False
        self.attn_c_max = 0
        self.cam_max = 3
        
    def build_img_dict(self, dataset):
        """
        build img dictionary from dataset
        # mnist
        imgs_dict = {
            number: {
                "imgs": tensor (number of img, 28, 28) 
                "index": LongTensor match to dataset
                }
            }
        # cifar10
        imgs_dict = {
            class: {
                "imgs": tensor (number of img, 28, 28) 
                "index": LongTensor match to dataset
                }
            }
        """
        self.class_to_idx = dataset.class_to_idx
        idx_to_class = {v:k for k, v in self.class_to_idx.items()}
        self.len_class = len(idx_to_class)
        self.class_datas_nums = []
        imgs_dict = {i:dict(imgs=[], index=[]) for i in range(self.len_class)}
        # mnist: img: ByteTensor (B, H, W) / targets: LongTensor
        # cifar10: img: np.array (B, H, W, C) / targets: list
        datas = dataset.data if isinstance(dataset.data, np.ndarray) else dataset.data.numpy()
        targets = dataset.targets if isinstance(dataset.targets, list) else dataset.targets.tolist()
        for i, (img, t) in enumerate(zip(datas, targets)):
            imgs_dict[t]["imgs"].append(torch.ByteTensor(img))
            imgs_dict[t]["index"].append(i)

        for i in range(self.len_class):
            imgs_dict[i]["imgs"] = torch.stack(imgs_dict[i]["imgs"]).numpy()
            self.class_datas_nums.append(len(imgs_dict[i]["index"]))
        return imgs_dict, idx_to_class

    def preprocessing(self, x, y):
        if not isinstance(x, torch.FloatTensor):
            x = self.test_dataset.transform(x).unsqueeze(0)
        if not isinstance(y, torch.LongTensor):
            y = torch.LongTensor([y])
        return x, y
    
    def create_model(self, m_type, a_type, del_p=None):
        m_class = self.model_dict[self.data_type][m_type]
        a_class = self.attr_dict[a_type]
        a_kwargs = self.get_kwargs_to_attr_model(self.data_type, m_type, a_type)
        if a_type in ["vanillagrad", "inputgrad", "guidedgrad"] and self.no_abs_grad:
            a_kwargs["abs_grad"] = False
        if del_p is None:
            load_path = str(self.sv_main_path/f"{m_type}-first.pt")
            attr_model = self.create_attr_model(m_class, a_class, load_path, a_kwargs)
            return attr_model
        else:
            p_text = f"{m_type}-{a_type}"
            if a_type in ["vanillagrad", "inputgrad", "guidedgrad"] and self.no_abs_grad:
                p_text += "-noabs"
            if self.fill_global_mean:
                p_text += "-fgm"
            p_text += f"-{del_p}.pt"
            load_path = str(self.sv_attr_path/p_text)
            model = m_class().eval()
            model.load_state_dict(torch.load(load_path, map_location="cpu"))
            return model

    def convert_sizes(self, x, byte=False):
        if x.size(1) == 3:
            x = x.squeeze(0).squeeze(0).permute(1, 2, 0)  # (1, C, H, W) > (H, W, C)
        else:
            x = x.squeeze(0).squeeze(0)  # (1, 1, H, W) > (H, W)
        if byte:
            return x.byte().numpy()
        else:
            return x.numpy()

    def forward(self, x, y, m_type, a_type, del_p, cam):
        if x.ndim <= 2:
            x = x[:, :, np.newaxis]  # (H, W, C)
        origin_img = x
        x, y = self.preprocessing(x, y)  # x: (1, C, H, W)
        # predict
        attr_model = self.create_model(m_type, a_type, del_p=None)
        output = attr_model.model(x)
        pred = output.argmax(dim=1).item()
        # get_attribution
        if a_type == "gradcam":
            attribution = attr_model.get_attribution(x, y, key=cam)
        else:
            attribution = attr_model.get_attribution(x, y)
        attribution = self.convert_scale(attribution)  # (1, C, H, W)

        masks = self.calculate_masks(attribution, del_p)  # (1, C, H, W)
        # masked x: (H, W, C) > (1, C, H, W) 
        masked_x = self.fill_datas_by_masks(torch.ByteTensor(origin_img).permute(2, 0, 1).unsqueeze(0), masks)
        # masked image predict
        model = self.create_model(m_type, a_type, del_p=del_p)
        masked_x_input = self.preprocessing(masked_x.squeeze(0).permute(1, 2, 0).numpy(), 0)[0]
        output = model(masked_x_input)
        masked_pred = output.argmax(dim=1).item()

        # TODO: generalize attention
        if m_type in ["resnetcbam", "resnetanr"]:
            if "cbam"in m_type:
                _ = attr_model.model.forward_map(masked_x_input)
                self.attention_dict = defaultdict(list)
                for k, v in attr_model.model.maps.items():
                    layer, typ = k.split("-")
                    self.attention_dict[layer].append(v.detach())
            else:
                _ = attr_model.model.forward_map(masked_x_input)
                self.attention_dict = attr_model.model.maps
            self.wg_attn.options = self.attention_dict.keys()
        else:
            self.attention_dict = self.attention_dict_default

        attribution = self.convert_sizes(attribution, byte=False)
        masked_img = self.convert_sizes(masked_x, byte=True)
        
        return pred, attribution, masked_img, masked_pred
        
    def create_widgets(self):
        style = {'description_width': 'initial'}
        self.wg_m_types = widgets.Dropdown(options=self.model_type, value=self.model_type[0], 
                                           description='Model:', disabled=False, style=style)
        self.wg_a_types = widgets.Dropdown(options=self.attr_type, value=self.attr_type[0], 
                                           description='Attribution:', disabled=False, style=style)
        self.wg_label = widgets.Dropdown(options=self.class_to_idx.keys(), 
                                         description='Label:', disabled=False, style=style)
        self.wg_index = widgets.IntSlider(description='Data Index', value=0, min=0, 
                                          max=self.class_datas_nums[self.class_to_idx.get(self.wg_label.value)]-1, style=style)
        self.wg_del_p = widgets.FloatSlider(description='delete/recover %', value=10.0, min=10.0, 
                                            max=90.0, step=10.0, readout_format=".0f", style=style)
        self.wg_cam = widgets.IntSlider(description='GradCAM Idx', value=0, min=0, max=self.cam_max, style=style)
        self.wg_attn = widgets.Dropdown(options=self.attention_dict.keys(), description='Attention:', disabled=False, style=style)
        self.wg_attn_c = widgets.IntSlider(description='Channel Select', value=0, min=0, max=self.attn_c_max, style=style)
        self.wg_reduce_dim = widgets.Checkbox(value=False, disabled=False, indent=False, style=style)
        rcd_box = widgets.Box([widgets.Label("Reduce Channel dim: ", style=style, layout=widgets.Layout(width="250px")), 
                            self.wg_reduce_dim], ayout=widgets.Layout(flex_flow="row", justify_content="flex-start"))

        # m_type | index   | cam 
        # a_type | del_p   | attn
        # label  | rcd_box | attn_c
        # Layout(flex='1 1 auto', width='auto')
        

        left_box = widgets.VBox([self.wg_m_types, self.wg_a_types, self.wg_label], 
                    layout=widgets.Layout(display='flex', flex_flow='column'))
        middle_box = widgets.VBox([self.wg_index, self.wg_del_p, rcd_box], 
                    layout=widgets.Layout(display='flex', flex_flow='column', width='320px'))
        right_box = widgets.VBox([self.wg_cam, self.wg_attn, self.wg_attn_c], 
                    layout=widgets.Layout(display='flex', flex_flow='column'))
        form = widgets.HBox([left_box, middle_box, right_box], 
            layout=widgets.Layout(display='inline-flex',flex_flow='row',border='solid 2px', justify_content='space-between'))
        interactive_dict = {"m_type": self.wg_m_types, "a_type": self.wg_a_types, "label": self.wg_label, 
                            "index": self.wg_index, "del_p": self.wg_del_p, "cam": self.wg_cam, 
                            "attn": self.wg_attn, "attn_c": self.wg_attn_c}
        return form, interactive_dict
        
    def draw_img(self, m_type, a_type, label, index, del_p, cam, attn, attn_c):
        label = self.class_to_idx[label]
        del_p /= 100
        img = self.img_dict[label]["imgs"][index]
        H, W = img.shape[0], img.shape[1]
        pred, attribution, masked_img, masked_pred = self.forward(img, label, m_type, a_type, del_p, cam)
        fig = plt.figure(figsize=(16, 10))

        fig_ax1 = fig.add_subplot(141)
        fig_ax1.imshow(img)

        fig_ax2 = fig.add_subplot(142)
        im = fig_ax2.imshow(attribution, cmap="rainbow")
        divider = make_axes_locatable(fig_ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        fig_ax3 = fig.add_subplot(143)
        fig_ax3.imshow(img)
        fig_ax3.imshow(attribution, cmap="coolwarm", alpha=0.6)
        
        fig_ax4 = fig.add_subplot(144)
        fig_ax4.imshow(masked_img)
        titles = [f"Predict: {self.idx_to_class.get(pred)}", "Attribution Map", "Model is Looking at",
                  f"Masked Image\nPredict: {self.idx_to_class.get(masked_pred)}"]
        for ax, title in zip([fig_ax1, fig_ax2, fig_ax3, fig_ax4], titles):
            ax.set_title(title)
            ax.axis("off")
        
        # draw attention
        fig2 = plt.figure(figsize=(14, 5), constrained_layout=False)
        if self.attention_dict.get(attn) is not None:            
            if "cbam" in m_type:
                gs = fig2.add_gridspec(nrows=3, ncols=6, hspace=0.1, wspace=0.4)
                f2_ax1 = fig2.add_subplot(gs[0, :])
                f2_ax2 = fig2.add_subplot(gs[1:, :2])
                f2_ax3 = fig2.add_subplot(gs[1:, 2:4])
                f2_ax4 = fig2.add_subplot(gs[1:, 4:6])

                # c_attn: (1, C, 1, 1), s_attn: (1, 1, H, W), out_attn: (1, C, H, W)
                c_attn, s_attn, out_attn = self.attention_dict.get(attn)
                s_attn_interpolated = F.interpolate(s_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
            
                im1 = f2_ax1.matshow(c_attn.squeeze().unsqueeze(0), cmap="rainbow")
                f2_ax1.yaxis.set_visible(False)
                f2_ax1.xaxis.set_ticks_position('bottom')
                f2_ax1.set_title(f"Channel Attentions (Total: {c_attn.size(1)})")
                
                im2 = f2_ax2.matshow(s_attn.squeeze(0).squeeze(0), cmap="rainbow")
                f2_ax2.set_title(f"Spatial Attentions")
                f2_ax2.xaxis.set_ticks_position('bottom')
                
                f2_ax3.imshow(img)
                f2_ax3.matshow(s_attn_interpolated.squeeze(0), cmap="rainbow", alpha=0.5)
                f2_ax3.set_title(f"Spatial Attentions(Interpolated)")
                f2_ax3.axis("off")
                
                if self.wg_reduce_dim.value:
                    self.wg_attn_c.max = 0
                    self.wg_attn_c.set_trait("value", 0)
                    t_out_attn = out_attn.mean(1).unsqueeze(1)  # (1, 1, H, W)
                else:
                    self.wg_attn_c.max = c_attn.size(1) - 1
                    # self.wg_attn_c.set_trait("value", 0)
                    s_idx = self.wg_attn_c.value
                    t_out_attn = out_attn[:, s_idx, :, :].unsqueeze(1)  # (1, 1, H, W)
                    
                im4 = f2_ax4.matshow(t_out_attn.squeeze(0).squeeze(0), cmap="rainbow")
                f2_ax4.set_title(f"Output after Attention")
                f2_ax4.xaxis.set_ticks_position('bottom')
                
                all_ims = [im1, im2, im4]
                all_axes = [f2_ax1, f2_ax2, f2_ax4]
                for i, (im_color, ax_color) in enumerate(zip(all_ims, all_axes)):
                    divider = make_axes_locatable(ax_color)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig2.colorbar(im_color, cax=cax)
                
            elif "anr" in m_type:

                gs = fig2.add_gridspec(nrows=1, ncols=2, wspace=0.5, left=0.02, right=0.7)
                f2_ax1 = fig2.add_subplot(gs[0, 0])
                f2_ax2 = fig2.add_subplot(gs[0, 1])

                attn_tensor = self.attention_dict.get(attn).detach()  # (1, C, H, W)
                # TODO: Spatial attention 2020.1.20
                # attention is calculated along the channel dimension 
                # so must show attention weight by channel
                # may plot like a 3 dim cube?
                # -------------- fix ----------------
                if self.wg_reduce_dim.value:
                    self.wg_attn_c.max = 0
                    self.wg_attn_c.set_trait("value", 0)
                    t_attn = attn_tensor.mean(1).unsqueeze(1)  # (1, 1, H, W)
                    t_attn_interpolated = F.interpolate(t_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
                    sub_title = "Collapsed"
                else:
                    self.wg_attn_c.max = attn_tensor.size(1) - 1
                    s_idx = self.wg_attn_c.value
                    t_attn = attn_tensor[:, s_idx, :, :].unsqueeze(1)  # (1, 1, H, W)
                    t_attn_interpolated = F.interpolate(t_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
                    sub_title = ""
                
                im1 = f2_ax1.imshow(t_attn.squeeze(0).squeeze(0), cmap="rainbow")
                f2_ax1.set_title(f"Attention Head {sub_title}")
                f2_ax1.xaxis.set_ticks_position('bottom')
                divider = make_axes_locatable(f2_ax1)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig2.colorbar(im1, cax=cax)
                
                f2_ax2.imshow(img)
                f2_ax2.imshow(t_attn_interpolated.squeeze(0), cmap="rainbow", alpha=0.5)
                f2_ax2.set_title(f"Attention Head {sub_title}(Interpolated)")
                f2_ax2.axis("off")
                # -------------- fix ----------------
                # all_ims = [im1, im2]
                # all_axes = [f2_ax1, f2_ax2]
                # for im_color, ax_color in zip(all_ims, all_axes):
                #     divider = make_axes_locatable(ax_color)
                #     cax = divider.append_axes("right", size="5%", pad=0.05)
                #     fig2.colorbar(im_color, cax=cax)
            plt.show()
        else:
            # self.wg_attn.set_trait("value", "none")
            plt.close(fig2)
            
    def show(self):
        form, interactive_dict = self.create_widgets()
        out = widgets.interactive_output(self.draw_img, interactive_dict)        
        display(form, out)

    def show_eval(self):
        with self.record_path.open() as f:
            x = f.read().splitlines()[2:]
            x = [line.strip("|").split("|") for line in x]
            m_type, a_type, percent, score = list(zip(*x))
            s_dict = defaultdict(list)
            for i in range(len(x)):
                s_dict[f"{m_type[i]}-{a_type[i]}"].append(np.float(score[i].strip("%")))

        percentages = np.arange(0, 100, 10).astype(np.float16)

        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in np.arange(len(set(a_type)))]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for i, (k, v) in enumerate(s_dict.items()):
            m, a = k.split("-")
            color_idx = i % len(set(a_type))
            if m == "resnet":
                axes[0].plot(percentages, np.array(v), label=a, color=colors[color_idx], marker="o")
            elif m == "resnetcbam":
                axes[1].plot(percentages, v, label=a, color=colors[color_idx], marker="o")
            else:
                axes[2].plot(percentages, v, label=a, color=colors[color_idx], marker="o")
        titles = ["resnet", "resnetcbam", "resnetanr"]
        for ax, title in zip(axes, titles):
            ax.grid(True)
            if self.data_type == "mnist":
                ax.set_yticks(np.arange(40.0, 110.0, 10))
            else:
                ax.set_yticks(np.arange(0.0, 110.0, 10))
            ax.set_xticks(percentages)
            if self.eval_type == "roar":
                ax.set_xlabel("% of deletion")
            else:
                ax.set_xlabel("% of recover")
            ax.set_title(title, fontsize=16)
            ax.legend()
        suptitle = self.record_path.name.split(".")[0]
        fig.suptitle(suptitle, fontsize=20, y=0.05)
        plt.show()