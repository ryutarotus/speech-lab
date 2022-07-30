import logging
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ttslearn.util import make_pad_mask
from .ttslearn.tacotron.attention import LocationSensitiveAttention
from .ttslearn.tacotron.encoder import Encoder
from .ttslearn.tacotron.postnet import Postnet
from .ttslearn.tacotron.decoder import Decoder, Prenet, ZoneOutCell, decoder_init
from .ttslearn.util import pad_1d

#from parallel_wavegan.layers import Conv1d
#from parallel_wavegan.layers import upsample
from .parallel_wavegan.layers import Conv1d1x1
from .parallel_wavegan.layers import WaveNetResidualBlock as ResidualBlock
#from parallel_wavegan import models

class Tacotron2(nn.Module):
    """Tacotron 2

    This implementation does not include the WaveNet vocoder of the Tacotron 2.

    Args:
        num_vocab (int): the size of vocabulary
        embed_dim (int): dimension of embedding
        encoder_hidden_dim (int): dimension of hidden unit
        encoder_conv_layers (int): the number of convolution layers
        encoder_conv_channels (int): the number of convolution channels
        encoder_conv_kernel_size (int): kernel size of convolution
        encoder_dropout (float): dropout rate of convolution
        attention_hidden_dim (int): dimension of hidden unit
        attention_conv_channels (int): the number of convolution channels
        attention_conv_kernel_size (int): kernel size of convolution
        decoder_out_dim (int): dimension of output
        decoder_layers (int): the number of decoder layers
        decoder_hidden_dim (int): dimension of hidden unit
        decoder_prenet_layers (int): the number of prenet layers
        decoder_prenet_hidden_dim (int): dimension of hidden unit
        decoder_prenet_dropout (float): dropout rate of prenet
        decoder_zoneout (float): zoneout rate
        postnet_layers (int): the number of postnet layers
        postnet_channels (int): the number of postnet channels
        postnet_kernel_size (int): kernel size of postnet
        postnet_dropout (float): dropout rate of postnet
        reduction_factor (int): reduction factor
    """

    def __init__(
        self,
        num_vocab=52,
        embed_dim=512,
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        attention_hidden_dim=128,
        attention_conv_channels=32,
        attention_conv_kernel_size=31,
        decoder_out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=2,
        spk_emb_dim=6,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_vocab,
            embed_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = Decoder(
            encoder_hidden_dim + spk_emb_dim,
            decoder_out_dim,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.postnet = Postnet(
            decoder_out_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )

    def forward(self, seq, in_lens, z, decoder_targets):
        """Forward step

        Args:
            seq (torch.Tensor): input sequence
            in_lens (torch.Tensor): input sequence lengths
            decoder_targets (torch.Tensor): target sequence

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        # エンコーダによるテキストの潜在表現の獲得
        encoder_outs = self.encoder(seq, in_lens)
        
        # 話者埋め込みを時間方向にexpandする
        z = z.unsqueeze(1)
        z = z.expand(z.shape[0], seq.shape[1], z.shape[-1])
        
        #print(encoder_outs.dtype) #float32
        #print(z.dtype) #float64
        
        decoder_inp = torch.cat((encoder_outs, z), dim=-1).to(torch.float32)
        
        #print(decoder_inp.dtype) #float64

        # デコーダによるメルスペクトログラム、stop token の予測
        outs, logits, att_ws = self.decoder(decoder_inp, in_lens, decoder_targets)

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs, outs_fine, logits, att_ws


    def inference(self, seq, z):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence

        Returns:
            tuple: tuple of outputs, outputs (after post-net), stop token prediction
                and attention weights.
        """
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)
        
        z = z.unsqueeze(0)

        outs, outs_fine, logits, att_ws = self.forward(seq, in_lens, z, None)

        return outs[0], outs_fine[0], logits[0], att_ws[0]
    
class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=30,
        stacks=3,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        aux_context_window=2,
        dropout=0.0,
        bias=True,
        use_weight_norm=True,
        use_causal_conv=False,
        upsample_conditional_features=True,
        upsample_net="ConvInUpsampleNetwork",
        upsample_params={"upsample_scales": [4, 5, 3, 5]},
    ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features:
            upsample_params.update(
                {
                    "use_causal_conv": use_causal_conv,
                }
            )
            if upsample_net == "MelGANGenerator":
                assert aux_context_window == 0
                upsample_params.update(
                    {
                        "use_weight_norm": False,  # not to apply twice
                        "use_final_nonlinear_activation": False,
                    }
                )
                self.upsample_net = getattr(models, upsample_net)(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update(
                        {
                            "aux_channels": aux_channels,
                            "aux_context_window": aux_context_window,
                        }
                    )
                self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # perform upsampling
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)

        # encode to hidden representation
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(
        layers, stacks, kernel_size, dilation=lambda x: 2 ** x
    ):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c=None, x=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(
                next(self.parameters()).device
            )
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            if normalize_before:
                c = (c - self.mean) / self.scale
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)