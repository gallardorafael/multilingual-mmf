# Copyright (c) Facebook, Inc. and its affiliates.
import functools

import torch
from torch import nn
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)
from .lighter_m4c import lighter_M4C

@registry.register_model("cnmt")
class CNMT(lighter_M4C):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")
        self.remove_unk_in_pred = self.config.remove_unk_in_pred

        # CNMT repetition mask
        repetition_mask = None
        coverage_ratio = -1e10

    @classmethod
    def config_path(cls):
        return "configs/models/cnmt/defaults.yaml"

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)

        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                "distilbert-base-cased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {"module": self.text_bert, "lr_scale": self.config.lr_scale_text_bert}
            )
        elif self.config.text_bert_init_from_bert_base_multilingual:
            self.text_bert = TextBert.from_pretrained(
                "distilbert-base-multilingual-cased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {"module": self.text_bert, "lr_scale": self.config.lr_scale_text_bert}
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_ocr_encoding(self):
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, "remove_ocr_fasttext", False
        )
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # 4) OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)
        self.linear_ocr_conf_to_mmt_in = nn.Linear(1, 768) # CNMT confidence

        # Layer norm for OCR features
        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_conf_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size) # CNMT confidence
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # 1) OCR FastText feature (300-dim) -> Now GloVe 300-d vectors, check GloVeProcessor class in processors.
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # 3) OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # 2) OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1
        )

        ocr_conf = sample_list.ocr_confidence # CNMT confidence
        ocr_bbox = sample_list.ocr_bbox_coordinates

        # linearly project each feature into d-dimensional space
        # then, sum them up as the final OCR token embedding
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
                self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
            ) + self.ocr_conf_layer_norm(
                self.linear_ocr_conf_to_mmt_in(ocr_conf) # CNMT confidence
            )
        )

        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_output(self, sample_list, fwd_results):
        super()._forward_output(sample_list, fwd_results)

        if self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

        return fwd_results

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results["prev_inds"] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds)
            fwd_results["prev_inds"][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for _ in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # CNMT repetition mask
                repetition_mask = torch.zeros(fwd_results["scores"].shape, device=fwd_results["scores"].device) # (32,30,6786)

                for sample_idx in range(fwd_results["scores"].shape[0]):
                    for step in range(fwd_results["scores"].shape[1]):
                        wd = fwd_results["prev_inds"][sample_idx][step]
                        if wd <= 20:    #common words
                            continue
                        repetition_mask[sample_idx, step+1:, wd] =  self.coverage_ratio
                        for same_wd in self.multiple_list[sample_idx][wd]:
                            repetition_mask[sample_idx, step+1:, same_wd] =  self.coverage_ratio

                fwd_results["scores"] = fwd_results["scores"] + repetition_mask

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]

def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output
