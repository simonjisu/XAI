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

        args = get_parser(data_type=wg_dataset_type, option=wg_eval_type, no_attention=True)
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
    
    wg_dict = {
        "m_type": wg_m_type, "a_type": wg_a_type, "label": wg_label, 
        "index": wg_index, "del_p": wg_del_p, "cam": wg_cam
    }
    fig1, _, eval_fig = explorer.streamlit_show(wg_dict)
    st.title("Attribution Method: Gradient")
    st.markdown("---")
    st.markdown(
    """
    gradient method includes:

    * [Vanilla Gradient](https://arxiv.org/abs/1312.6034)
    * [Input Gradient](https://arxiv.org/abs/1611.07634)
    * [Guided Gradient](https://arxiv.org/abs/1412.6806)
    * [Grad CAM](https://arxiv.org/abs/1610.02391)

    Basically, given a input image $X$ and the target label $Y$, the attribution map is built by 
    gradient of input $X$. 

    """)
    
    st.header("Experiment")
    st.markdown(
    """
    - Dealing with negative gradients: Are negative gradients important in attributions? 
    - How does the color dimension of attribution map effect to image? Compairing the two training result:
        1. do not reduce for color dimension imformation of attribution map, ranking all pixcel information including channels.
        2. gray-scale the attribution map (reduce color dimension to 1)
    - Fill color with black(0.0) versus mean of global channels(gray)
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
    """)
    st.subheader(f"Qualitative: Attribution Maps")
    st.pyplot(fig1)
    st.subheader("Quantitative: Test accuracy changes")
    _, col_center, _ = st.beta_columns([1, 2, 1])
    col_center.pyplot(eval_fig)
    st.markdown(
    """
    In some of experiment the accuracy goes up. This means that some attribution method cannot catch the 
    important pixel for classification
    """)
