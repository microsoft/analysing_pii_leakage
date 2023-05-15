from transformers import GPT2Config

from src.models.language_model import LanguageModel


class GPT2(LanguageModel):
    """ A custom convenience wrapper around huggingface gpt-2 utils """

    def get_config(self):
        return GPT2Config()


