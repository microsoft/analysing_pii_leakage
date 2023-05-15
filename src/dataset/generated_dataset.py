import os

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.ner_args import NERArgs
from src.arguments.sampling_args import SamplingArgs
from src.dataset.dataset import Dataset
from src.global_configs import system_configs


class GeneratedDataset(Dataset):

    def __init__(self, model_args: ModelArgs, sampling_args: SamplingArgs, ner_arg: NERArgs = None, env_args: EnvArgs = None):
        """ A generated dataset is identified by the dataset args and sampling args. """
        super().__init__(ner_args=ner_arg, env_args=env_args)
        self.sampling_args = sampling_args
        self.model_args = model_args

    @property
    def _pii_cache(self):
        """ Returns the filepath for the file that contains all pii and their location. """
        return os.path.join(os.path.abspath(system_configs.CACHE_DIR), f"{self.model_args.hash(suffix='pii')}_{self.sampling_args.hash(suffix='pii')}")

    def _load_base_dataset(self):
        raise NotImplementedError


