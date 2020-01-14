import torch
import logging
import time
from distutils.version import LooseVersion
from sacred import Experiment
from easydict import EasyDict as edict
from logging.config import dictConfig
from pccgan.utils import random_init, check_para_correctness, create_logger

from pccgan.models import PCCGenerator


ex = Experiment()

@ex.command
def train(_run, _rnd, _seed):

    # random_init(_seed)
    cfg = edict(_run.config)
    random_init(cfg.seed)

    check_para_correctness(cfg)
    ex.logger = create_logger(cfg, postfix='_train')
    
    
    classifer = KWSClassifier(cfg=cfg,
                               logger=ex.logger)
    classifer.train()

@ex.command
def test(_run, _rnd, _seed):

    # random_init(_seed)
    cfg = edict(_run.config)
    random_init(cfg.seed)
    check_para_correctness(cfg)
    ex.logger = create_logger(cfg, postfix='_test')
    
    cfg.debug = True
    classifer = KWSClassifier(cfg=cfg,
                               logger=ex.logger)
    classifer.test_only()


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.1'), \
        'PyTorch>=0.4.1 is required'

    ex.add_config('./experiment/default.yaml')
    ex.run_commandline()