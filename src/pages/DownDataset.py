import streamlit as st
from ..trainsettings import ModelTranier
from ..parserutils import get_parser
from pathlib import Path

def check_dataset_exist(args):
    # check if already exists
    if args.data_type == "mnist":
        data_folder = "MNIST"
    elif args.data_type == "cifar10":
        data_folder = "cifar-10-batches-py"
    else:
        raise ValueError("Wrong data type")
    download = False if (Path(args.data_path) / data_folder).exists() else True
    return download

def download_dataset(eval_type_options):
    for data_type in ["mnist", "cifar10"]:
        option = eval_type_options[data_type][0]
        args = get_parser(data_type=data_type, option=option, no_attention=True)
        download = check_dataset_exist(args)
        if download:
            args.download = download
            with st.spinner(f"Downloading Dataset ..."):
                trainer = ModelTranier()
                _ = trainer.build_dataset(args)