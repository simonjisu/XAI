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
                   help="Attribution Method Type: deconv, gradcam, guidedgrad, relavance, vanillagrad, inputgrad, guided_gradcam. \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-at deconv gradcam`")
    parser.add_argument("-mt", "--model_type", nargs="+",
                   help="Model Type: cnn, resnet, resnetcbam, resnetanr \
                        Insert at least one method, some attribution method will not be supported to some models. \
                        example:`-mt resnet resnetcbam resentanr`")
    # attribution details
    parser.add_argument("-rcd", "--reduce_color_dim", type=str, default=None,
                   help="Reduce the color channel of dimension in the attribution maps by following methods.\
                        Methods: mean, rec601, itu_r_bt707, itu_r_bt2100")
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
    parser.add_argument("-skip2","--skip_second_eval", action="store_true",
                   help="Skip second evaluation")
    
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
        self.reduce_color_dim = args.reduce_color_dim
        self.prj_path = Path(args.prj_path)
        if self.reduce_color_dim is not None:
            path_txt = f"trained/{self.data_type}/{self.eval_type}/{self.reduce_color_dim}"
        else:
            path_txt = f"trained/{self.data_type}/{self.eval_type}"
        self.p = self.prj_path / path_txt
        
        _, self.test_dataset, *_ = self.build_dataset(args)
        self.img_dict, self.idx_to_class = self.build_img_dict(self.test_dataset)
        # attenion options
        self.attention_dict_default = {"none": None}
        self.attention_dict = self.attention_dict_default
        self.attn_option = False
        self.attn_c_max = 0
        self.cam_max = 5
        
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
        if del_p is None:
            load_path = str(self.p/f"{m_type}-first.pt")
            _, attr_model = self.create_attr_model(m_class, a_class, load_path, a_kwargs)
            return attr_model
        else:
            load_path = str(self.p/f"{m_type}-{a_type}-{del_p}.pt")
            model = m_class().eval()
            model.load_state_dict(torch.load(load_path, map_location="cpu"))
            return model
            
    def forward(self, x, y, m_type, a_type, del_p, cam):
        origin_img = x
        x, y = self.preprocessing(x, y)
        # predict
        attr_model = self.create_model(m_type, a_type, del_p=None)
        output = attr_model.model(x)
        pred = output.argmax(dim=1).item()
        # get_attribution
        if a_type == "gradcam":
            attribution = attr_model.get_attribution(x, y, key=cam)
        else:
            attribution = attr_model.get_attribution(x, y)
        attribution, C = self.convert_scale(attribution, C=attribution.size(1))
        if del_p != 0.0:
            masks = self.calculate_masks(attribution, del_p)  # (1, C, H, W)
            # masked x: (H, W, C) > (C, H, W) 
            masked_x = torch.ByteTensor(origin_img).permute(2, 0, 1).masked_fill(masks.squeeze(0), 0)
            # masked image predict
            model = self.create_model(m_type, a_type, del_p=del_p)
            masked_x_input = self.preprocessing(masked_x.permute(1, 2, 0).numpy(), 0)[0]
            output = model(masked_x_input)
            masked_pred = output.argmax(dim=1).item()
            masked_img = masked_x.permute(1, 2, 0).numpy()
        else:
            masks = torch.zeros_like(attribution, dtype=torch.bool)
            masked_img = origin_img
            masked_pred = None
        # TODO: generalize attention
        if m_type in ["resnetcbam", "resnetanr"]:
            if "cbam"in m_type:
                if del_p != 0.0:
                    _ = attr_model.model.forward_map(masked_x_input)
                else:
                    _ = attr_model.model.forward_map(x)
                self.attention_dict = defaultdict(list)
                for k, v in attr_model.model.maps.items():
                    layer, typ = k.split("-")
                    self.attention_dict[layer].append(v.detach())
            else:
                if del_p != 0.0:
                    _ = attr_model.model.forward_map(masked_x_input)
                else:
                    _ = attr_model.model.forward_map(x)
                self.attention_dict = attr_model.model.maps
            self.wg_attn.options = self.attention_dict.keys()
        else:
            self.attention_dict = self.attention_dict_default            
        if C == 3:
            # (1, C, H, W) > (1, H, W, C)
            masked_img = masked_x.squeeze().permute(1, 2, 0)
            attribution = attribution.permute(0, 2, 3, 1)
        
        attribution = attribution.squeeze().numpy()
        masks = masks.squeeze().numpy()

        return pred, attribution, masks, masked_img, masked_pred
        
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
        self.wg_del_p = widgets.FloatSlider(description='Delete %', value=0.0, min=0.0, 
                                            max=90.0, step=10.0, readout_format=".0f", style=style)
        self.wg_cam = widgets.IntSlider(description='GradCAM Idx', value=0, min=0, max=self.cam_max, style=style)
        self.wg_attn = widgets.Dropdown(options=self.attention_dict.keys(), description='Attention:', disabled=False, style=style)
        self.wg_attn_c = widgets.IntSlider(description='Channel Select', value=0, min=0, max=self.attn_c_max, style=style)
        self.wg_reduce_dim = widgets.Checkbox(value=False, disabled=False, indent=False, style=style)
        # form_item_layout = widgets.Layout(
        #     display='flex',
        #     flex_flow='column',
        # )
        # m_type | index   | cam 
        # a_type | del_p   | attn
        # label  | rcd_box | attn_c
        # Layout(flex='1 1 auto', width='auto')
        rcd_box = widgets.Box([widgets.Label("Whether Reduce Channel dimension: ", style=style), self.wg_reduce_dim], 
                                layout=widgets.Layout(flex_flow="row", justify_content="space-between"))
        left_box = widgets.VBox([self.wg_m_types, self.wg_a_types, self.wg_label], 
                    layout=widgets.Layout(
                        display='flex',
                        flex_flow='column',
                    ))
        middle_box = widgets.VBox([self.wg_index, self.wg_del_p, rcd_box], 
                    layout=widgets.Layout(
                        display='flex',
                        flex_flow='column',
                        width='320px'
                    ))
        right_box = widgets.VBox([self.wg_cam, self.wg_attn, self.wg_attn_c], 
                    layout=widgets.Layout(
                        display='flex',
                        flex_flow='column',
                    ))
        form = widgets.HBox([left_box, middle_box, right_box], layout=widgets.Layout(
            display='inline-flex',
            flex_flow='row',
            border='solid 2px',
            justify_content='space-between',
            # width='100%'
        ))
        interactive_dict = {"m_type": self.wg_m_types, "a_type": self.wg_a_types, "label": self.wg_label, 
                            "index": self.wg_index, "del_p": self.wg_del_p, "cam": self.wg_cam, 
                            "attn": self.wg_attn, "attn_c": self.wg_attn_c}
        return form, interactive_dict
        
    def draw_img(self, m_type, a_type, label, index, del_p, cam, attn, attn_c):
        label = self.class_to_idx[label]
        del_p /= 100
        img = self.img_dict[label]["imgs"][index]
        H, W, _ = img.shape
        pred, attribution, masks, masked_img, masked_pred = self.forward(img, label, m_type, a_type, del_p, cam)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 10))
        titles = [f"Predict: {self.idx_to_class.get(pred)}", 
                  "Attribution Map", "Mask",
                  f"Masked Image\nPredict: {self.idx_to_class.get(masked_pred)}"]
        plotimgs = [img, attribution, masks, masked_img]
        for i, (ax, im, title)in enumerate(zip(axes, plotimgs, titles)):
            if i == 1:
                im_color = ax.imshow(im, cmap="rainbow")
                # attribution maps
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im_color, cax=cax)
            elif i == 2:
                ax.imshow(im, cmap="YlGn")
            else:
                ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
        
        # draw attention
        fig2 = plt.figure(figsize=(14, 5), constrained_layout=False)
        if self.attention_dict.get(attn) is not None:            
            if "cbam" in m_type:
                gs = fig2.add_gridspec(nrows=2, ncols=8, hspace=0.02, wspace=0.8)
                f2_ax1 = fig2.add_subplot(gs[0, :])
                f2_ax2 = fig2.add_subplot(gs[1, :2])
                f2_ax3 = fig2.add_subplot(gs[1, 2:4])
                f2_ax4 = fig2.add_subplot(gs[1, 4:6])
                f2_ax5 = fig2.add_subplot(gs[1, 6:])

                # c_attn: (1, C, 1, 1), s_attn: (1, 1, H, W), out_attn: (1, C, H, W)
                c_attn, s_attn, out_attn = self.attention_dict.get(attn)
                s_attn_interpolated = F.interpolate(s_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
            
                f2_ax1.matshow(c_attn.squeeze().unsqueeze(0), cmap="rainbow")
                f2_ax1.yaxis.set_visible(False)
                f2_ax1.xaxis.set_ticks_position('bottom')
                f2_ax1.set_title(f"Channel Attentions (Total: {c_attn.size(1)})")
                
                im2 = f2_ax2.matshow(s_attn.squeeze(0).squeeze(0), cmap="rainbow")
                f2_ax2.set_title(f"Spatial Attentions")
                f2_ax2.xaxis.set_ticks_position('bottom')
                
                im3 = f2_ax3.matshow(s_attn_interpolated.squeeze(0), cmap="rainbow")
                f2_ax3.set_title(f"Spatial Attentions(Interpolated)")
                f2_ax3.axis("off")
                
                if self.wg_reduce_dim.value:
                    self.wg_attn_c.max = 0
                    self.wg_attn_c.set_trait("value", 0)
                    t_out_attn = out_attn.mean(1).unsqueeze(1)  # (1, 1, H, W)
                    t_out_attn_interpolated = F.interpolate(t_out_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
                else:
                    self.wg_attn_c.max = c_attn.size(1) - 1
                    s_idx = self.wg_attn_c.value
                    t_out_attn = out_attn[:, s_idx, :, :].unsqueeze(1)  # (1, 1, H, W)
                    t_out_attn_interpolated = F.interpolate(t_out_attn, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
                    
                im4 = f2_ax4.matshow(t_out_attn.squeeze(0).squeeze(0), cmap="rainbow")
                f2_ax4.set_title(f"Output after Attention")
                f2_ax4.xaxis.set_ticks_position('bottom')

                im5 = f2_ax5.matshow(t_out_attn_interpolated.squeeze(0), cmap="rainbow")
                f2_ax5.set_title(f"Output after Attention(Interpolated)")
                f2_ax5.axis("off")
                
                all_ims = [im2, im3, im4, im5]
                all_axes = [f2_ax2, f2_ax3, f2_ax4, f2_ax5]
                for im_color, ax_color in zip(all_ims, all_axes):
                    divider = make_axes_locatable(ax_color)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig2.colorbar(im_color, cax=cax)
                
            elif "anr" in m_type:
                gs = fig2.add_gridspec(nrows=1, ncols=2, wspace=0.5, left=0.02, right=0.7)
                f2_ax1 = fig2.add_subplot(gs[0, 0])
                f2_ax2 = fig2.add_subplot(gs[0, 1])

                attn_tensor = self.attention_dict.get(attn).detach()  # (1, C, H, W)
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

                im2 = f2_ax2.imshow(t_attn_interpolated.squeeze(0), cmap="rainbow")
                f2_ax2.set_title(f"Attention Head {sub_title}(Interpolated)")
                f2_ax2.axis("off")
                
                all_ims = [im1, im2]
                all_axes = [f2_ax1, f2_ax2]
                for im_color, ax_color in zip(all_ims, all_axes):
                    divider = make_axes_locatable(ax_color)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig2.colorbar(im_color, cax=cax)
            plt.show()
        else:
            self.wg_attn.set_trait("value", "none")
            plt.close(fig2)
            
            # self.attention_dict = self.attention_dict_default
            # self.wg_attn.options = self.attention_dict.keys()
            # self.wg_attn.set_trait("value", "none")
            
    def show(self):
        form, interactive_dict = self.create_widgets()
        out = widgets.interactive_output(self.draw_img, interactive_dict)        
        display(form, out)