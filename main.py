__author__ = "simonjisu"

from pathlib import Path
import sys

from models.utils import argument_parsing
from models.trainsettings import ModelTranier

if __name__ == "__main__":
    args = argument_parsing()
    print(args)
    trainer = ModelTranier()
    trainer.main(args)