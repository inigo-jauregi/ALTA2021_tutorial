# Script to train the language model for text classification
# Script to train the joint model
import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForSequenceClassification, \
    BartForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
from sacrebleu.metrics import BLEU

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from jtt.evaluation_metrics import translation_metrics


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_prefix, tgt_prefix, tokenizer, max_input_len, max_output_len):

        src_examples = []
        with open(hf_dataset + '.' + src_prefix) as src_reader:
            for line in src_reader:
                src_examples.append(tokenizer.encode(line.strip()))

        tgt_examples = []
        with open(hf_dataset + '.' + tgt_prefix) as tgt_reader:
            for line in tgt_reader:
                tgt_examples.append(tokenizer.encode(line.strip()))

        if len(src_examples) != len(tgt_examples):
            raise "Number of source and target sentences must be the same."

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
        # print(src_sen)
        # tgt_sen = [self.tokenizer.decode(t, skip_special_tokens=False) for t in entry['tgt_ids']]
        # print(tgt_sen)

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


class LmForTranslation(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
        self.tokenizer.save_pretrained(self.args.save_dir, self.args.save_prefix, 'tokenizer')
        self.model = BartForConditionalGeneration.from_pretrained(self.args.model_lm_path)

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

        # self.writer = open('lightning_preds.txt', 'w')
        # self.writer_label = open('lightning_labels.txt', 'w')

    def forward(self, src_ids, src_attention_mask, tgt_ids, tgt_attention_mask):
        labels = tgt_ids[:, 1:].clone()
        decoder_input_ids = tgt_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(
            src_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        lm_logits = outputs.logits
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size  # It has to be the same as the output class
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )

        # Metrics
        # acc = translation_metrics(lm_logits.detach(), labels)
        acc = torch.tensor(0.8)

        return [loss, acc]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss.detach(), 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(
                                loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0,
                            'accuracy': output[1]}
        self.log("my_lr", lr, prog_bar=True, on_step=True)
        self.log("accuracy", output[1], prog_bar=True, on_step=True)
        return {'loss': loss, 'accuracy': output[1], 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        # Generate translation
        src_ids, src_attention_mask, tgt_ids, tgt_attention_mask = batch
        generated_ids = self.model.generate(input_ids=src_ids, attention_mask=src_attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=1)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(tgt_ids.tolist(), skip_special_tokens=True)
        # TODO: Potentially detokenize???

        return {'vloss': vloss, 'vaccuracy': outputs[1], 'ref_sentences': gold_str, 'pred_sentences': generated_str}

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'vaccuracy']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.accelerator_connector.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        # Calculate BLEU score
        names += ['BLEU']
        ref_sentences = []
        pred_sentences = []
        for x in outputs:
            ref_sentences += x['ref_sentences']
            pred_sentences += x['pred_sentences']
        bleu_scorer = BLEU()
        bleu_score = bleu_scorer.corpus_score(pred_sentences, ref_sentences)
        logs = dict(zip(*[names, metrics]))
        self.log("BLEU", bleu_score.score, prog_bar=True, on_step=True)
        print(logs)

        # Save language model
        self.save_language_model(bleu_score.score)

        return {'avg_val_loss': logs['vloss'], 'avg_accuracy': logs['vaccuracy'], 'log': logs, 'progress_bar': logs}

    def save_language_model(self, bleu_score):
        path_save = os.path.join(self.args.save_dir, self.args.save_prefix, "checkpoints_" + str(bleu_score))
        self.model.save_pretrained(path_save, True)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.debug:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args.dataset_size * self.args.epochs / num_gpus / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = TranslationDataset(hf_dataset=self.hf_datasets[split_name], src_prefix=self.args.src,
                                     tgt_prefix=self.args.tgt, tokenizer=self.tokenizer,
                                     max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  shuffle=is_train) if \
            self.trainer.accelerator_connector.use_ddp else None
        # Shuffle or not
        if is_train and (sampler is None):
            is_shuffle = True
        else:
            is_shuffle = False
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_shuffle,
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=TranslationDataset.collate_fn)

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'validation', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--train_data", type=str, required=True, help='Path to training data')
        parser.add_argument("--validation_data", type=str, required=True, help='Path to validation data')
        parser.add_argument("--test_data", type=str, required=True, help='Path to testing data')
        parser.add_argument("--src", type=str, required=True, help='Source language.')
        parser.add_argument("--tgt", type=str, required=True, help='Target language.')
        parser.add_argument("--save_dir", type=str, default='translation')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--max_grad_norm", type=float, default=1.0, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=0,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Adam weight decay")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_input_len", type=int, default=170,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_output_len", type=int, default=170,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_lm_path", type=str, default='../pretrained_lm/sshleifer-tiny-mbart',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='../pretrained_lm/sshleifer-tiny-mbart')
        parser.add_argument("--progress_bar", type=int, default=10, help="Progress bar. Good for printing")
        parser.add_argument("--precision", type=int, default=32, help="Double precision (64), full precision (32) "
                                                                      "or half precision (16). Can be used on CPU, "
                                                                      "GPU or TPUs.")
        parser.add_argument("--amp_backend", type=str, default='native', help="The mixed precision backend to "
                                                                              "use ('native' or 'apex')")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.from_pretrained is not None:
        model = LmForTranslation.load_from_checkpoint(args.from_pretrained, args)
    else:
        model = LmForTranslation(args)

    model.hf_datasets = {'train': args.train_data,
                         'validation': args.validation_data,
                         'test': args.test_data}
    print(model.hf_datasets)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=1,
        verbose=True,
        monitor='BLEU',
        mode='max',
        period=0
    )

    print(args)

    args.dataset_size = 203037  # hardcode dataset size. Needed to compute number of steps for the lr scheduler

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         gradient_clip_val=args.max_grad_norm,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         logger=logger,
                         callbacks=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=args.progress_bar,
                         precision=args.precision,
                         amp_backend=args.amp_backend, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="translation_model")
    parser = LmForTranslation.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
