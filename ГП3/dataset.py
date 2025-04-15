from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
from tokenizers.processors import TemplateProcessing
from tqdm.auto import tqdm
import torch

def train_tokenizer(texts):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=50000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=1)
    tokenizer.train_from_iterator(tqdm(texts, desc="Training tokenizer"), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B [EOS]",
        special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ])
    return tokenizer


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe['text'].tolist()
        self.summaries = dataframe['summary'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.tokenizer.encode(self.texts[idx]).ids)
        summary = torch.tensor(self.tokenizer.encode(self.summaries[idx]).ids)
        return text, summary


def collate_fn(batch, max_len=512):
    texts, summaries = zip(*batch)

    def pad_to_max(sequence, max_length, pad_value=1):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return torch.cat([sequence, torch.full((max_length - len(sequence),), pad_value)], dim=0)

    texts_padded = torch.stack([pad_to_max(text, 449) for text in texts])
    summaries_padded = torch.stack([pad_to_max(summary, 95) for summary in summaries])

    return texts_padded, summaries_padded