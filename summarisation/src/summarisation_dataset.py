import torch
from torch.utils.data import DataLoader, Dataset


class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, max_output_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        input_ids = self.tokenizer.encode(entry['document'], truncation=True, max_length=self.max_input_len)
        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=self.max_output_len)

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        if batch[0][0][-1].item() == 2:
            pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        elif batch[0][0][-1].item() == 1:
            pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        else:
            assert False

        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids
