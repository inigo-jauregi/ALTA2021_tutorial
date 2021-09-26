from transformers import AutoTokenizer, BartForConditionalGeneration

# Lightning modules
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class LmForTranslation(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams['params'] = params
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer)
        self.tokenizer.save_pretrained(self.args.save_dir, self.args.save_prefix)
        self.model = BartForConditionalGeneration.from_pretrained(self.args.model_lm_path)

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

        # self.writer = open('lightning_preds.txt', 'w')
        # self.writer_label = open('lightning_labels.txt', 'w')

    def forward(self, src_ids, src_attention_mask, tgt_ids, tgt_attention_mask):
        labels = tgt_ids[:, 1:].clone()
        # print([self.tokenizer.decode(t, skip_special_tokens=False) for t in labels])
        decoder_input_ids = tgt_ids[:, :-1]
        # print([self.tokenizer.decode(t, skip_special_tokens=False) for t in decoder_input_ids])
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        outputs = self.model(
            src_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False
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
        acc = translation_accuracy(lm_logits.detach(), labels, tokenizer=self.tokenizer)

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
        src_str = self.tokenizer.batch_decode(src_ids.tolist(), skip_special_tokens=True)
        generated_ids = self.model.generate(input_ids=src_ids, attention_mask=src_attention_mask,
                                            decoder_start_token_id=self.tokenizer.bos_token_id,
                                            max_length=self.args.max_output_len,
                                            num_beams=1, num_beam_groups=1, do_sample=False)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        gold_str = self.tokenizer.batch_decode(tgt_ids.tolist(), skip_special_tokens=True)

        return {'vloss': vloss, 'vaccuracy': outputs[1], 'ref_sentences': gold_str, 'pred_sentences': generated_str}

    def validation_epoch_end(self, outputs, test=False):
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
                # print(x['ref_sentences'])
                # ref_sentences += x['ref_sentences']
            for pred in x['pred_sentences']:
                pred_sentences.append(pred)
                # print(x['pred_sentences'])
                # pred_sentences += x['pred_sentences']
        bleu_scorer = BLEU()
        bleu_score = bleu_scorer.corpus_score(pred_sentences, ref_sentences)
        logs = dict(zip(*[names, metrics]))
        self.log("BLEU", bleu_score.score, prog_bar=True)
        # print(logs)

        # Save language model
        if test is not True:
            self.save_language_model(bleu_score.score)

        return {'avg_val_loss': logs['vloss'], 'avg_accuracy': logs['vaccuracy'], 'log': logs, 'progress_bar': logs}

    def save_language_model(self, bleu_score):
        path_save = os.path.join(self.args.save_dir, self.args.save_prefix, "checkpoints_" + str(bleu_score))
        self.model.save_pretrained(path_save, True)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs, test=True)
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
        parser.add_argument("--model_lm_path", type=str, default='../pretrained_lms/sshleifer-tiny-mbart',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='../pretrained_lms/sshleifer-tiny-mbart')
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