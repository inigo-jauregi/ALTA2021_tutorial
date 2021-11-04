# Script to train a summarisation model with Pytorch Lightning and
# a pretrainedSeq2seq model from Huggingface

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
from rouge_score import rouge_scorer

import pytorch_lightning as pl

from src.summarisation_dataset import SummarizationDataset
from torch.nn.parallel import DistributedDataParallel


class LmForSummarisation(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['tokenizer'], use_fast=True)

        config = AutoConfig.from_pretrained(self.args['model_path'])
        config.attention_dropout = self.args['attention_dropout']
        if self.args['model_path'] == 'allenai/led-base-16384':
            config.gradient_checkpointing = self.args['grad_ckpt']
            config.attention_mode = self.args['attention_mode']
            config.attention_window = [self.args['attention_window']] * config.encoder_layers
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args['model_path'], config=config)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        print(config)

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        # attention_mask[:, 0] = 2  # global attention on one token for all model params to be used (impt for grad ckpt)
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        labels = output_ids[:, 1:].clone()
        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,)
        lm_logits = outputs[0]
        if self.args['label_smoothing'] == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = self.label_smoothed_nll_loss(
                lprobs, labels, self.args['label_smoothing'], ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]

    def summarise_example(self, input_document):
        # Tokenize the document
        input_ids = self.tokenizer.encode(input_document, truncation=True, max_length=self.args['max_input_len'])
        input_ids = torch.tensor(input_ids)
        # Generate attention mask
        # input_ids, attention_mask = self._prepare_input(input_ids)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        # attention_mask[:, 0] = 2

        doc_ids = torch.nn.utils.rnn.pad_sequence([input_ids], batch_first=True, padding_value=1)
        doc_attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(attention_mask)],
                                                             batch_first=True, padding_value=0)

        # decoder_start_token_id = self.tokenizer.bos_token_id
        generated_ids = self.model.generate(input_ids=doc_ids, attention_mask=doc_attention_mask,
                                            use_cache=True, max_length=self.args['max_output_len'],
                                            num_beams=1)
        # Decode to string
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
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args['max_output_len'],
                                            num_beams=1)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        for ref, pred in zip(gold_str, generated_str):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_str)
        rouge2 /= len(generated_str)
        rougel /= len(generated_str)
        rougelsum /= len(generated_str)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum, }

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.accelerator_connector.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        self.log("validation_loss", logs['vloss'], prog_bar=True)
        self.log("average_rouge", (logs['rouge1']+logs['rouge2']+logs['rougeL'])/3, prog_bar=True)
        print(logs)
        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)

    def configure_optimizers(self):
        if self.args['adafactor']:
            optimizer = Adafactor(self.model.parameters(), lr=self.args['lr'], scale_parameter=False,
                                  relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])
        if self.args['debug']:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args['dataset_size'] * self.args['epochs'] / num_gpus / self.args['grad_accum'] / \
                    self.args['batch_size']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args['warmup'], num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        dataset = SummarizationDataset(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
                                       max_input_len=self.args['max_input_len'],
                                       max_output_len=self.args['max_output_len'])
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if \
            self.trainer.accelerator_connector.use_ddp else None
        return DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=(sampler is None),
                          num_workers=self.args['num_workers'], sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

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
