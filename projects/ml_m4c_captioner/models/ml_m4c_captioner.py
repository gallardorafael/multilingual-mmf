# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from .ml_m4c import ml_M4C


@registry.register_model("ml_m4c_captioner")
class M4CCaptioner(ml_M4C):
    def __init__(self, config):
        super().__init__(config)
        self.remove_unk_in_pred = self.config.remove_unk_in_pred

    @classmethod
    def config_path(cls):
        return "configs/models/ml_m4c_captioner/defaults.yaml"

    def _forward_output(self, sample_list, fwd_results):
        super()._forward_output(sample_list, fwd_results)

        if self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

        return fwd_results
