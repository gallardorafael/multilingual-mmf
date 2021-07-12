# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from omegaconf import OmegaConf
from .lighter_m4c import lighter_M4C


@registry.register_model("lighter_m4c_captioner")
class M4CCaptioner(lighter_M4C):
    def __init__(self, config):
        super().__init__(config)
        self.remove_unk_in_pred = self.config.remove_unk_in_pred

    @classmethod
    def config_path(cls):
        return "configs/models/lighter_m4c_captioner/defaults.yaml"

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

    def _forward_output(self, sample_list, fwd_results):
        super()._forward_output(sample_list, fwd_results)

        if self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

        return fwd_results
