import streamlit as st
from pathlib import Path
from PIL import Image
from .DownDataset import check_dataset_exist, download_dataset
from ..parserutils import get_parser
from ..utils import Explorer

def write():
    repo_path = Path(".").absolute()
    assets_path = repo_path / "assets"
    eval_type_options = {
        "mnist": ["roar", "kar"],
        "cifar10": ["roar-plain", "kar-rcd", "roar-rcd", "roar-rcd-fgm", "roar-rcd-noabs"]
    }

    st.title("Attribution Method: Additional Experiments")
    st.markdown("---")
    
    st.header("1. Dealing with negative gradients")
    st.markdown(
    """
    When using ResNet model, found that without getting absolute negative gradients, 
    some gradient attribution method will increase the model accuracy. This may caused by under reason.

    **The negative gradients** can also be important features for image, when ranking the attribuion, 
    the big negative gradients will be ranked lower. Making the import features are not deleted from 
    the image, means that the background is deleted from the image, resulting the model to classify 
    more accurately.

    You can find some lower attribution scores(blue) are coexist with important objects. However, 
    the deleted part is background of the image.
    """)

    img1 = Image.open(assets_path / "exp_cifar10_roar_rcd_absVSnoabs.png")
    st.image(img1, caption="Compare the ROAR result between absoluting negative gradients and not")

    st.header("2. Collasping the color dimension")
    st.markdown(
    """
    For color image, there is color dimension for each pixel. I tested collasping the color dimension 
    by convert the attribution map to gray-scale.

    * how to change color image to gray-scale? https://en.wikipedia.org/wiki/Grayscale

    There is little different between collasping color dimension of attribution map and not collasping on test results. 
    But attribution map is not human readable.
    """)

    img2 = Image.open(assets_path / "exp_cifar10_roar_rcd_plainVSrcd.png")
    st.image(img2, caption="Compare the ROAR result between gray scaling the attribution map or not")

    st.header("3. Fill different colors to masks")
    st.markdown(
    """
    Guided gradient works better filling with the global mean of pixels than filling with the black color.
    """)
    
    img3 = Image.open(assets_path / "exp_cifar10_roar_rcd_mask-blackVSglobalmean.png")
    st.image(img3, caption="Compare the ROAR result filling the mask with between global-mean and black")

    