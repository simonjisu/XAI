import streamlit as st
from pathlib import Path
from PIL import Image
from ..parserutils import get_parser
from ..utils import Explorer

def write():
    repo_path = Path(".").absolute()
    assets_path = repo_path / "assets"
    eval_type_options = {
        "mnist": ["roar", "kar"],
        "cifar10": ["roar-plain", "kar-rcd", "roar-rcd", "roar-rcd-fgm", "roar-rcd-noabs"]
    }
            
    sidebar_container = st.sidebar.beta_container()
    with sidebar_container:
        st.markdown("**Arguments**")
        wg_dataset_type = st.selectbox("Select a Dataset: ", options=["mnist", "cifar10"], index=1)
        wg_eval_type = st.selectbox("Select a Evaluation Type with options:", 
            options=eval_type_options[wg_dataset_type], index=0)

        args = get_parser(data_type=wg_dataset_type, option=wg_eval_type, no_attention=False)
        explorer = Explorer(args)
        
        st.markdown("---")
        wg_m_type = st.selectbox("Model: ", options=explorer.model_type, index=0)
        wg_a_type = st.selectbox("Attribution Method: ", options=explorer.attr_type, index=0)
        wg_label = st.selectbox("Label: ", options=list(explorer.class_to_idx.keys()), index=0)
        wg_index = st.slider("Data Index: ", value=0, min_value=0, 
            max_value=explorer.class_datas_nums[explorer.class_to_idx.get(wg_label)]-1)
        del_p_str = "Delete" if "roar" in wg_eval_type else "Recover"
        wg_del_p = st.slider(f"{del_p_str}: ", value=10.0, min_value=0.0, max_value=90.0, step=10.0)
        if wg_a_type == "gradcam":
            wg_cam = st.slider("GradCAM Idx: ", value=0, min_value=0, max_value=explorer.cam_max)
        else:
            wg_cam = 0
        
        # forward random value to get attention_dict
        if wg_m_type in ["resnetcbam", "resnetanr"]:
            attn_c_str = "Channel Attention: " if wg_m_type == "resnetcbam" else "Attention Head: "
            reduce_dim_str = "Output after Attention" if wg_m_type == "resnetcbam" else "Attention Head"
            explorer.forward_random(data_type=wg_dataset_type, m_type=wg_m_type, a_type=wg_a_type)
            wg_attn = st.selectbox("Attention Layers: ", options=list(explorer.attention_dict.keys()), index=0)
            wg_reduce_dim = st.checkbox(f"Mean Channel in '{reduce_dim_str}'", value=False)
            if wg_reduce_dim:
                wg_attn_c = 0
            else:
                wg_attn_c = st.slider(attn_c_str, value=0, min_value=0, max_value=explorer.attn_c_max)
        else:
            wg_attn = 0
            wg_attn_c = 0
            wg_reduce_dim = False
    
    wg_dict = {
        "m_type": wg_m_type, "a_type": wg_a_type, "label": wg_label, 
        "index": wg_index, "del_p": wg_del_p, "cam": wg_cam,
        "attn": wg_attn, "attn_c": wg_attn_c, "reduct_dim": wg_reduce_dim
    }
    fig1, fig2, eval_fig = explorer.streamlit_show(wg_dict)


    st.title("Attribution Method: Attention")
    st.markdown("---")
    st.markdown(
    """
    Attention method includes:

    * [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
    * [Attend and Rectify: a Gated Attention Mechanism for Fine-Grained Recovery](https://arxiv.org/abs/1807.07320)

    """)
    st.header("Experiment")
    st.markdown(
    """
    1. Dealing with negative gradients: Are negative gradients important in attributions? 
    2. How does the color dimension of attribution map effect to image? Compairing the two training result:
        1. do not reduce for color dimension imformation of attribution map, ranking all pixcel information including channels.
        2. gray-scale the attribution map (reduce color dimension to 1)
    3. Fill color with black(0.0) versus mean of global channels(gray)
    """)
    with st.beta_expander("👉 See what are the options mean"):
        st.markdown(
        """
        * `plain`: basic setting - absoulte attribution scores
        * `rcd`: gray scale for all attribution methods(means that reducing the color dimension to 1)
        * `fgm`: fill the masks with global mean of all datas instead of zeros.
        * `noabs`: not to absolute attribution scores in some methods
        """)
    st.header("Experiment Results")
    st.markdown(
    """
    1. Qualitative: See Attribution Maps with Mnist / Cifar10 test dataset
    2. Quantitative: watch test accuracy in ROAR/KAR is decreasing / increasing.

    Builded a Baseline model which create a attribution map by random values from [0, 1).

    * all attributions are normalized to 0 to 255.
    * In KAR evaluation, percent of recover at 0 means the test accuracy for model's first training result.
    """)
    st.subheader(f"Qualitative: Attribution Maps")
    st.pyplot(fig1)
    if fig2 is not None:
        st.pyplot(fig2)
    st.subheader("Quantitative: Test accuracy changes")
    st.pyplot(eval_fig)
    st.markdown(
    """
    In some of experiment the accuracy goes up. This means that some attribution method cannot catch the 
    important pixel for classification
    """)


# rensetcbam - gradcam airplane data index 265, shows the model is looking at sky