# encoding: utf-8


import argparse
import os
from collections import namedtuple
from typing import Dict

import sys
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD
import numpy as np
np.set_printoptions(threshold=np.inf)

from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import collate_to_max_length
from metrics.query_span_f1 import QuerySpanF1
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig
from loss import *
from utils.get_parser import get_parser
from utils.radom_seed import set_random_seed
import logging

set_random_seed(0)


class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         mrc_dropout=args.mrc_dropout)

        self.model = BertQueryNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config)

        # logging.info(str(self.model))
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        # self.ce_loss = CrossEntropyLoss(reduction="none")
        self.loss_type = args.loss_type
        # self.loss_type = "bce"
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=args.dice_smooth)
        # todo(yuxian): 由于match loss是n^2的，应该特殊调整一下loss rate
        weight_sum = args.weight_start + args.weight_end + args.weight_span
        self.weight_start = args.weight_start / weight_sum
        self.weight_end = args.weight_end / weight_sum
        self.weight_span = args.weight_span / weight_sum
        self.flat_ner = args.flat
        self.span_f1 = QuerySpanF1(flat=self.flat_ner)
        self.chinese = args.chinese
        self.optimizer = args.optimizer
        self.span_loss_candidates = args.span_loss_candidates
        # self.type2id={1:'Model',2: 'Missile',3: 'method', 4:'parameter',5:'Math', 6:'phenomenon',
        #  7:'process'}
        self.type2id = {1: 'Model', 2: 'Missile', 3: 'method', 4: 'parameter', 5: 'phenomenon',
                        6: 'process'}
        self.id2pos = {1: 47, 2:10, 3: 38, 4: 30, 5: 24,
                        6: 19}
        self.tokenizer=BertWordPieceTokenizer(os.path.join(self.bert_dir, "vocab.txt"))
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="bert dropout rate")
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_span", type=float, default=1.0)
        parser.add_argument("--flat", action="store_true", help="is flat ner")
        parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "gold"],
                            default="all", help="Candidates used to compute span loss")
        parser.add_argument("--chinese", action="store_true",
                            help="is chinese dataset")
        parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                            help="loss type")
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                            help="loss type")
        parser.add_argument("--dice_smooth", type=float, default=1e-8,
                            help="smooth value of dice loss")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        start_outputs_param_optimizer = list(self.model.start_outputs.named_parameters())
        end_outputs_param_optimizer = list(self.model.end_outputs.named_parameters())
        span_embedding_param_optimizer = list(self.model.span_embedding.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.lr},
            {'params': [p for n, p in start_outputs_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.span_lr},
            {'params': [p for n, p in start_outputs_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.span_lr},
            {'params': [p for n, p in end_outputs_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr':self.args.span_lr},
            {'params': [p for n, p in end_outputs_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.span_lr},
            {'params': [p for n, p in span_embedding_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.span_match_lr},
            {'params': [p for n, p in span_embedding_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.args.span_match_lr}
        ]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        if self.loss_type == "bce":
            start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
            match_loss = match_loss * float_match_label_mask
            match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        else:
            start_loss = self.dice_loss(start_logits, start_labels.float(), start_float_label_mask)
            end_loss = self.dice_loss(end_logits, end_labels.float(), end_float_label_mask)
            match_loss = self.dice_loss(span_logits, match_labels.float(), float_match_label_mask)

        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""

        output = {}

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch

        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss

        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"match_loss"] = match_loss

        start_preds, end_preds = start_logits > 0, end_logits > 0
        start_label_mask = start_label_mask.bool()
        end_label_mask = end_label_mask.bool()
        match_labels = match_labels.bool()
        bsz, seq_len = start_label_mask.size()
        match_preds = span_logits > 0
        # [bsz, seq_len]
        start_preds = start_preds.bool()
        # [bsz, seq_len]
        end_preds = end_preds.bool()

        match_preds = (match_preds
                       & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                       & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                            & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
        match_preds = match_label_mask & match_preds
        match_preds_list = []
        batch_size, seq_len = batch[0].size()


        for k in range(batch_size):
            entity_type=self.type2id[label_idx[k].cpu().item()]
            query_pos = self.id2pos[label_idx[k].cpu().item()]
            for i in range(seq_len):
                for j in range(seq_len):
                    if match_preds[k, i, j]:
                        tokens_text = self.tokenizer.decode(tokens[k].cpu().numpy().tolist())
                        entity_text = self.tokenizer.decode(tokens[k][i:j+1].cpu().numpy().tolist())
                        new_token_text=''.join(tokens_text.split(' '))
                        new_entity_text = ''.join(entity_text.split(' '))

                        # for m in range(0,len(tokens_text),2):
                        #     new_token_text+=tokens_text[m]
                        match_preds_list.append([tokens[k],tokens_text,new_token_text[query_pos:], [i, j],new_entity_text,entity_type])
        output["match_preds_list"] = match_preds_list
        #print(start_preds.cpu().numpy(),match_labels.cpu().numpy())
        span_f1_stats = self.span_f1(start_preds=start_preds, end_preds=end_preds, match_logits=span_logits,
                                     start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                     match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats
        # batch_size, seq_len = batch[0].size()
        for idx in range(batch_size):
            typeid=label_idx[idx]
            start_pred=start_preds[idx].unsqueeze(0)
            end_pred=end_preds[idx].unsqueeze(0)
            span_logit=span_logits[idx].unsqueeze(0)
            one_start_label_mask=start_label_mask[idx].unsqueeze(0)
            one_end_label_mask=end_label_mask[idx].unsqueeze(0)
            match_label=match_labels[idx].unsqueeze(0)
            one_span_f1_stats=self.span_f1(start_preds=start_pred, end_preds=end_pred, match_logits=span_logit,
                                     start_label_mask=one_start_label_mask, end_label_mask=one_end_label_mask,
                                     match_labels=match_label)
            entity_type=self.type2id[typeid.item()]
            f1_name=entity_type+'_span_f1_stats'
            output[f1_name]=one_span_f1_stats
        return output

    def validation_epoch_end(self, outputs):
        """"""


        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_match_list=[]
        print(outputs)
        for x in outputs:
            for y in x['match_preds_list']:
                all_match_list.append(y)
        all_result={}
        for x in all_match_list:
            if x[2] not in all_result:
                all_result[x[2]]=[x[0],[[x[3],x[4],x[5]]]]
            else:
                all_result[x[2]][1].append([x[3],x[4],x[5]])
        write_path=os.path.join(self.args.default_root_dir,'pred_result.txt')
        with open(write_path,'w',encoding='utf-8')as f:
            for x in all_result:
                f.write('text:'+x+'\n')
                for y in all_result[x][1]:
                    f.write('entity_pos:'+str(y[0][0])+','+str(y[0][1])+'  ——  entity_text:'+y[1]+'  ——  entity_type:'+y[2]+'\n')
        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        out = sys.stdout
        out.write('\n')
        for idx in self.type2id:
            entity_type=self.type2id[idx]
            f1_name=entity_type+'_span_f1_stats'
            data_needed=[]
            for x in outputs:
                if f1_name in x:
                    data_needed.append(x[f1_name])
            if not data_needed:
                continue
            type_all_counts = torch.stack(data_needed).sum(0)
            type_span_tp, type_span_fp, type_span_fn = type_all_counts
            type_span_recall = type_span_tp / (type_span_tp + type_span_fn + 1e-10)
            type_span_precision = type_span_tp / (type_span_tp + type_span_fp + 1e-10)
            type_span_f1 = type_span_precision * type_span_recall * 2 / (type_span_recall + type_span_precision + 1e-10)
            p_name = entity_type + '_span_precision'
            q_name = entity_type + '_span_recall'
            f1_name1 = entity_type + '_span_f1'


            out.write('%17s: ' % entity_type)
            out.write('precision: %6.2f%%; ' % (100. * type_span_precision))
            out.write('recall: %6.2f%%; ' % (100. * type_span_recall))
            out.write('F1: %6.2f%% \n' % (100. *type_span_f1))
            # print(p_name+':'+str(type_span_precision.item())+'\t'+
            #              q_name+':'+str(type_span_recall.item())+'\t'+
            #              f1_name1+':'+str(type_span_f1.item())+'\t')
            tensorboard_logs[p_name] = type_span_precision
            tensorboard_logs[q_name] = type_span_recall
            tensorboard_logs[f1_name1] = type_span_f1

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")
        # return self.get_dataloader("dev", 100)

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")
        # return self.get_dataloader("dev")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=BertWordPieceTokenizer(vocab_path),
                                max_length=self.args.max_length,
                                is_chinese=self.chinese,
                                pad_to_maxlen=False
                                )

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader


def run_dataloader():
    """test dataloader"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args.workers = 0
    args.default_root_dir = "/mnt/data/mrc/train_logs/debug"

    model = BertLabeling(args)
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_config_dir, "vocab.txt"))

    loader = model.get_dataloader("dev", limit=1000)
    for d in loader:
        input_ids = d[0][0].tolist()
        match_labels = d[-1][0]
        start_positions, end_positions = torch.where(match_labels > 0)
        start_positions = start_positions.tolist()
        end_positions = end_positions.tolist()
        if not start_positions:
            continue
        print("="*20)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        for start, end in zip(start_positions, end_positions):
            print(tokenizer.decode(input_ids[start: end+1]))


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLabeling(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])
        print('ok load')

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=10,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=8
    )

    trainer.fit(model)
    result=trainer.test(model)
    print('all finished')


if __name__ == '__main__':
    # run_dataloader()
    main()
