from ..model import DeconvNet, GradCAM, VanillaGrad, InputGrad, GuidedGrad, LRP
from pathlib import Path

class XaiTrainer(object):
    def __init__(self):
        """
        Must implement following functions
        - train
        - test
        - main_train
        - build_dataset
        - main
        """
        self.attr_dict = {
            "deconv": DeconvNet, 
            "gradcam": GradCAM, 
            "guidedgrad": GuidedGrad, 
            "guided_gradcam": None,  # Not implemented yet
            "relavance": LRP, 
            "vanillagrad": VanillaGrad, 
            "inputgrad": InputGrad,
        }

    def train(self, *args):
        raise NotImplementedError

    def test(self, *args):
        raise NotImplementedError

    def main_train(self, *args):
        """
        contains `self.train` and `self.test` function
        """
        raise NotImplementedError

    def build_dataset(self, *args):
        raise NotImplementedError

    def main(self, *args):
        raise NotImplementedError

    def check_path_exist(self, path, directory=True):
        assert isinstance(path, Path), "path should be `pathlib.Path` type"
        if directory:
            if not path.exists():
                path.mkdir(parents=True)
                print(f"Given path doesn't exists, created {path}")
        else:
            if not path.exists():
                path.touch()
                print(f"Given path doesn't exists, created {path}")


