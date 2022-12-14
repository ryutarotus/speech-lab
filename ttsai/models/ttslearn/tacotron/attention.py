import torch
from torch import nn
from torch.nn import functional as F

def make_pad_mask(lengths, maxlen=None):
    """Make mask for padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask

class LocationSensitiveAttention(nn.Module):
    """Location-sensitive attention

    This is an attention mechanism used in Tacotron 2.

    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
        conv_channels (int): number of channels of convolutional layer
        conv_kernel_size (int): size of convolutional kernel
    """

    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.mlp_att = nn.Linear(conv_channels, hidden_dim, bias=False)
        assert conv_kernel_size % 2 == 1
        self.loc_conv = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None


    def forward(
        self,
        encoder_outs,
        src_lens,
        decoder_state,
        att_prev,
        mask=None,
    ):
        """Forward step

        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            att_prev (torch.FloatTensor): previous attention weight
            mask (torch.FloatTensor): mask for padding
        """
        # ????????????????????????????????????????????????????????????
        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(encoder_outs)

        # ???????????????????????????????????????????????????
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

        # (B, T_enc) -> (B, 1, T_enc) -> (B, conv_channels, T_enc) ->
        # (B, T_enc, conv_channels)
        att_conv = self.loc_conv(att_prev.unsqueeze(1)).transpose(1, 2)
        # (B, T_enc, hidden_dim)
        att_conv = self.mlp_att(att_conv)

        # (B, 1, hidden_dim)
        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        # NOTE: ?????????????????????????????????????????????????????????????????????????????????????????????2 ?????????????????????
        # 1) ???????????????????????????????????????????????????????????????
        # 2) ??????????????????????????????
        erg = self.w(
            torch.tanh(att_conv + self.processed_memory + decoder_state)
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # ??????????????????????????????????????????????????????????????????????????????
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights