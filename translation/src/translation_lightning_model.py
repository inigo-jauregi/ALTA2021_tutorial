# Script to train a translation model with Pytorch Lightning and
# a pretrainedSeq2seq model from Huggingface

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartTokenizer
from transformers.adapters.configuration import AdapterConfig
from transformers.optimization import get_linear_schedule_with_warmup

from sacrebleu.metrics import BLEU

import pytorch_lightning as pl

from src.translation_metrics import translation_accuracy
from src.translation_dataset import TranslationDataset


class LmForTranslation(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['tokenizer'], use_fast=False)
        if isinstance(self.tokenizer, MBartTokenizer):
            src_lan = self.args['src'] + "_XX"
            tgt_lan = self.args['tgt'] + "_XX"
            self.tokenizer._src_lang = src_lan
            self.tokenizer.tgt_lang = tgt_lan

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args['model'])

        if self.args['add_adapter']:
            config = AdapterConfig.load("pfeiffer", non_linearity="relu",
                                        reduction_factor=self.args['reduction_factor'])
            self.model.add_adapter("mt_adapter", config=config)
            self.model.train_adapter("mt_adapter")

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    def forward(self, src_ids, src_attention_mask, tgt_ids, tgt_attention_mask):

        labels = tgt_ids[:, 1:].clone()
        decoder_input_ids = tgt_ids[:, :-1]

        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(
            src_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False
        )
        lm_logits = outputs.logits
        if self.args['label_smoothing'] == 0:
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size  # It has to be the same as the output class
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = self.label_smoothed_nll_loss(
                lprobs, labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
            )

        # Metrics
        acc = translation_accuracy(lm_logits.detach(), labels, tokenizer=self.tokenizer)

        return [loss, acc]

    def translate_example(self, sentence):

        sen_ids = self.tokenizer.encode(sentence.strip())
        sen_attention_mask = [1] * len(sen_ids)

        sen_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(sen_ids)], batch_first=True, padding_value=1)
        sen_attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(sen_attention_mask)],
                                                             batch_first=True, padding_value=0)

        if isinstance(self.tokenizer, MBartTokenizer):
            decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]
        else:
            decoder_start_token_id = self.tokenizer.bos_token_id
        generated_ids = self.model.generate(input_ids=sen_ids, attention_mask=sen_attention_mask,
                                            decoder_start_token_id=decoder_start_token_id,
                                            max_length=self.args['max_tgt_len'],
                                            num_beams=1, num_beam_groups=1, do_sample=False)

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)

        return generated_str

    @staticmethod
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
        if isinstance(self.tokenizer, MBartTokenizer):
            decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]
        else:
            decoder_start_token_id = self.tokenizer.bos_token_id
        generated_ids = self.model.generate(input_ids=src_ids, attention_mask=src_attention_mask,
                                            decoder_start_token_id=decoder_start_token_id,
                                            max_length=self.args['max_tgt_len'],
                                            num_beams=1, num_beam_groups=1, do_sample=False)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(tgt_ids.tolist(), skip_special_tokens=True)

        return {'vloss': vloss, 'vaccuracy': outputs[1], 'ref_sentences': gold_str, 'pred_sentences': generated_str}

    def validation_epoch_end(self, outputs, test=False):
        if self.args['add_adapter']:
            for name, p in self.model.named_parameters():
                if 'adapters' in name:
                    p.requires_grad = True
        else:
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
        ref_sentences = [[]]
        pred_sentences = []
        for x in outputs:
            for ref in x['ref_sentences']:
                ref_sentences[0].append(ref)
            for pred in x['pred_sentences']:
                pred_sentences.append(pred)
        bleu_scorer = BLEU()
        bleu_score = bleu_scorer.corpus_score(pred_sentences, ref_sentences)
        logs = dict(zip(*[names, metrics]))
        self.log("BLEU", bleu_score.score, prog_bar=True)

        return {'avg_val_loss': logs['vloss'], 'avg_accuracy': logs['vaccuracy'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs, test=True)
        print(result)

    def configure_optimizers(self):
        # Define AdamW optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'],
                                      weight_decay=self.args['weight_decay'])

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(self.args)
        num_steps = self.args['dataset_size'] * \
                    self.args['epochs'] / num_gpus / self.args['grad_accum'] / self.args['batch_size']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args['warmup'], num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = TranslationDataset(hf_dataset=self.hf_datasets[split_name], src_prefix=self.args['src'],
                                     tgt_prefix=self.args['tgt'], tokenizer=self.tokenizer,
                                     max_input_len=self.args['max_src_len'],
                                     max_output_len=self.args['max_tgt_len'])
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  shuffle=is_train) if \
            self.trainer.accelerator_connector.use_ddp else None
        # Shuffle or not
        if is_train and (sampler is None):
            is_shuffle = True
        else:
            is_shuffle = False
        return DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=is_shuffle,
                          num_workers=1, sampler=sampler, collate_fn=TranslationDataset.collate_fn)

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
