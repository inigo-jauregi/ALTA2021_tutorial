import torch
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_prefix, tgt_prefix, tokenizer, max_input_len, max_output_len):

        src_examples = []
        with open(hf_dataset + '.' + src_prefix) as src_reader:
            for line in src_reader:
                src_examples.append(tokenizer.encode(line.strip()))

        tgt_examples = []
        with tokenizer.as_target_tokenizer():
            with open(hf_dataset + '.' + tgt_prefix) as tgt_reader:
                for line in tgt_reader:
                    tgt_examples.append(tokenizer.encode(line.strip()))

        if len(src_examples) != len(tgt_examples):
            raise ValueError

        examples = []
        for i in range(len(src_examples)):
            if len(src_examples[i]) <= max_input_len and len(tgt_examples[i]) <= max_output_len:
                examples.append({'src_ids': src_examples[i], 'tgt_ids': tgt_examples[i]})

        self.hf_dataset = examples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # self.writer_input = open('lightning_input.txt', 'w')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]

        # src_sen = [self.tokenizer.decode(t, skip_special_tokens=False) for t in entry['src_ids']]
        # print(entry['src_ids'])
        # print(src_sen)
        # tgt_sen = [self.tokenizer.decode(t, skip_special_tokens=False) for t in entry['tgt_ids']]
        # print(entry['tgt_ids'])
        # print(tgt_sen)
        # print(self.tokenizer.bos_token_id)
        # print(self.tokenizer.pad_token_id)

        src_attention_mask = [1] * len(entry['src_ids'])
        tgt_attention_mask = [1] * len(entry['tgt_ids'])

        # if self.tokenizer.bos_token_id is None:  # pegasus
        #     output_ids = [self.tokenizer.pad_token_id] +
        return torch.tensor(entry['src_ids']), torch.tensor(src_attention_mask), \
               torch.tensor(entry['tgt_ids']), torch.tensor(tgt_attention_mask)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        # if batch[0][0][-1].item() == 2:
        #     pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        # elif batch[0][0][-1].item() == 1:
        #     pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        # else:
        #     assert False

        src_ids, src_attention_mask, tgt_ids, tgt_attention_mask = list(zip(*batch))
        src_ids = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=1)
        src_attention_mask = torch.nn.utils.rnn.pad_sequence(src_attention_mask, batch_first=True, padding_value=0)
        tgt_ids = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=1)
        tgt_attention_mask = torch.nn.utils.rnn.pad_sequence(tgt_attention_mask, batch_first=True, padding_value=0)

        return src_ids, src_attention_mask, tgt_ids, tgt_attention_mask
