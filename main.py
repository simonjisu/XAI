__author__ = "simonjisu"

from pathlib import Path
import sys

from src.parserutils import argument_parsing
from src.trainsettings import ModelTranier

if __name__ == "__main__":
    args = argument_parsing()
    print(args)
    trainer = ModelTranier()
    trainer.main(args)