"""Wrapper around TTGAN model."""

import os
import pandas as pd
import torch

from ttgan.ttgan import TTGAN
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns


class TTGANSynthesizer(BaseSingleTableSynthesizer):
    """Model wrapping ``TTGAN`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original TTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool or str):
            If ``True``, use CUDA. If a ``str``, use the indicated device.
            If ``False``, do not use cuda at all.
    """

    _model_sdtype_transformers = {'categorical': None}

    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 target=None, embedding_dim=128, generator_num_layers=6, discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, checkpoint=None):

        super().__init__(
            metadata=metadata,
            enforce_min_max_values=enforce_min_max_values,
            enforce_rounding=enforce_rounding,
            locales=locales
        )

        self.target = target
        self.embedding_dim = embedding_dim
        self.generator_num_layers = generator_num_layers
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac
        self.cuda = cuda
        self.checkpoint = checkpoint

        self._model_kwargs = {
            'target': target,
            'embedding_dim': embedding_dim,
            'generator_num_layers': generator_num_layers,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'checkpoint': checkpoint,
        }

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data)
        self._model = TTGAN(**self._model_kwargs)
        self._model.fit(processed_data, discrete_columns=discrete_columns)

    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if conditions is None:
            return self._model.sample(num_rows)

        raise NotImplementedError("TTGANSynthesizer doesn't support conditional sampling.")


class TTGANWrapper:
    """Wrapper around ``TTGANSynthesizer`` model.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        target (str):
            Name of the target column.
    """

    def __init__(self, metadata, target, verbose, epochs, checkpoint):
        self.target     = target
        self.model_A    = TTGANSynthesizer(metadata, target="A", verbose=verbose, epochs=epochs, checkpoint=checkpoint)
        self.model_B    = TTGANSynthesizer(metadata, target="B", verbose=verbose, epochs=epochs, checkpoint=checkpoint)

    def fit(self, data):
        """Fit the model to the table."""
        self.model_A.fit(data[data[self.target] == 0])
        self.model_B.fit(data[data[self.target] == 1])

    def sample(self, num_rows):
        """Sample the indicated number of rows from the model."""
        return pd.concat([
            self.model_A.sample(num_rows // 2),
            self.model_B.sample(num_rows // 2),
        ]).sample(frac=1).reset_index(drop=True)