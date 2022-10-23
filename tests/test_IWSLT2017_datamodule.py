from pathlib import Path

import pytest
import torch

from src.datamodules.IWSLT2017_datamodule import IWSLT2017DataModule


@pytest.mark.parametrize(("batch_size", "language_pair", "vocab_path", "spm_model_path"), [(32, ("en", "de"), "https://download.pytorch.org/models/text/xlmr.vocab.pt", "https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model")])
def test_IWSLT2017_datamodule(batch_size, language_pair, vocab_path, spm_model_path):
    data_dir = "data/"

    dm = IWSLT2017DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        language_pair=language_pair,
        spm_model_path=spm_model_path,
        vocab_path=vocab_path
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "IWSLT2017").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
