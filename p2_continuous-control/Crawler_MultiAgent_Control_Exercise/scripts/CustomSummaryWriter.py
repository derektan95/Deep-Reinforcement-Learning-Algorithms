import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

class CustomSummaryWriter(SummaryWriter):
    """
    Overrides SummaryWriter class simply to remove duplicate creation of subdirs when 
    add_scalar & add_hparams are called separately.
    https://github.com/pytorch/pytorch/issues/32651       

    Simply removes the portion 'str(time.time())' 
    https://github.com/pytorch/pytorch/blob/8072f0685f5bd9bc8f1e48ef916518fa31a50826/torch/utils/tensorboard/writer.py#L306
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)