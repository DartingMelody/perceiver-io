import torch
import torch.nn as nn

from einops import repeat
from tokenizers import Tokenizer
from typing import Tuple

from perceiver.adapter import (
    InputAdapter,
    OutputAdapter
)
from perceiver.tokenizer import (
    UNK_TOKEN,
    MASK_TOKEN,
    SPECIAL_TOKENS
)
from perceiver.utils import Sequential


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels)
    )


def cross_attention_layer(num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float):
    return Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout)
    )


def self_attention_layer(num_channels: int, num_heads: int, dropout: float):
    return Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout)
    )


def self_attention_block(num_layers: int, num_channels: int, num_heads: int, dropout: float):
    return Sequential(*[self_attention_layer(num_channels, num_heads, dropout) for _ in range(num_layers)])


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=num_q_channels,
                                               num_heads=num_heads,
                                               kdim=num_kv_channels,
                                               vdim=num_kv_channels,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    # Simplified version of cross-attention module described in https://arxiv.org/abs/2103.03206.
    # Here, the embedding dimension is determined by the number of query channels (num_q_channels)
    # whereas in the paper it can be specified separately. This simplification allows re-use of the
    # torch.nn.MultiHeadAttention module whereas a full implementation of the paper would require a
    # custom multi-head attention implementation.
    def __init__(self,
                 num_q_channels: int,
                 num_kv_channels: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(num_q_channels=num_q_channels,
                                            num_kv_channels=num_kv_channels,
                                            num_heads=num_heads,
                                            dropout=dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(num_q_channels=num_channels,
                                            num_kv_channels=num_channels,
                                            num_heads=num_heads,
                                            dropout=dropout)

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class PerceiverEncoder(nn.Module):
    def __init__(self,
                 input_adapter: InputAdapter,
                 latent_shape: Tuple[int, int],
                 num_layers: int,
                 num_cross_attention_heads: int = 4,
                 num_self_attention_heads: int = 4,
                 num_self_attention_layers_per_block: int = 2,
                 dropout: float = 0.0):
        """
        Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to an encoder input of shape
                              (B, M, C_input) where B is the batch size, M the input sequence length and C_input
                              the number of input channels.
        :param latent_shape: Shape of the latent array, (N, C_latent), where N is the number of latent variables
                             and C_latent the number of latent channels.
        :param num_layers: Number of encoder layers. An encoder layer is composed of a cross-attention layer and
                           several self-attention layers (= a self-attention block).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param dropout: Dropout for self- and cross-attention layers and residuals.
        """
        super().__init__()

        self.input_adapter = input_adapter
        self.num_layers = num_layers

        num_latent_channels = latent_shape[1]

        def create_perceiver_layer():
            return Sequential(
                cross_attention_layer(num_latent_channels,
                                      input_adapter.num_input_channels,
                                      num_cross_attention_heads,
                                      dropout),
                self_attention_block(num_self_attention_layers_per_block,
                                     num_latent_channels,
                                     num_self_attention_heads,
                                     dropout)
            )

        self.layer_1 = create_perceiver_layer()

        if num_layers > 1:
            # will be used recurrently depending on num_layers
            self.layer_n = create_perceiver_layer()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(*latent_shape))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, '... -> b ...', b=b)

        x_latent = self.layer_1(x_latent, x, pad_mask)
        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(self,
                 output_adapter: OutputAdapter,
                 latent_shape: Tuple[int, int],  # as produced by perceiver encoder
                 num_cross_attention_heads: int = 4,
                 dropout: float = 0.0):
        """
        Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder output of shape (B, K, C_output) to task-specific
                               output. B is the batch size, K the output sequence length and C_output the
                               number of output channels. (K, C_output) is specified via the output_shape
                               property of the output_adapter.
        :param latent_shape: Shape of the latent array, (N, C_latent), as producer by a Perceiver IO encoder,
                             where N is the number of latent variables and C_latent the number of latent channels.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param dropout: Dropout for cross-attention layers and residuals.
        """
        super().__init__()

        num_latent_channels = latent_shape[1]
        num_output_channels = output_adapter.output_shape[-1]

        self.output_adapter = output_adapter
        self.latent_shape = latent_shape
        self.cross_attention = cross_attention_layer(num_q_channels=num_output_channels,
                                                     num_kv_channels=num_latent_channels,
                                                     num_heads=num_cross_attention_heads,
                                                     dropout=dropout)

        self.output = nn.Parameter(torch.empty(*output_adapter.output_shape))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.latent_shape:
            raise ValueError(f'Latent shape {tuple(d)} different from required shape {self.latent_shape}')

        output = repeat(self.output, '... -> b ...', b=b)
        output = self.cross_attention(output, x)
        return self.output_adapter(output)


class TextMasking(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 unk_token_id: int,
                 mask_token_id: int,
                 num_special_tokens: int,
                 mask_p: float = 0.15):
        """
        Text masking as described in https://arxiv.org/abs/1810.04805.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.num_special_tokens = num_special_tokens
        self.mask_p = mask_p

    @staticmethod
    def create(tokenizer: Tokenizer, **kwargs):
        return TextMasking(vocab_size=tokenizer.get_vocab_size(),
                           unk_token_id=tokenizer.token_to_id(UNK_TOKEN),
                           mask_token_id=tokenizer.token_to_id(MASK_TOKEN),
                           num_special_tokens=len(SPECIAL_TOKENS),
                           **kwargs)

    def forward(self, x, pad_mask):
        labels = x.clone()

        # Mask special tokens in input (UNK, PAD)
        is_special = x == self.unk_token_id
        is_special |= pad_mask

        # Mask non-special tokens
        is_input = ~is_special

        # Randomly select 15% of non-special tokens
        is_selected = torch.rand_like(x, dtype=torch.float) < self.mask_p
        is_selected &= is_input

        # Of those, set 80% to MASK token, 10% to random token and leave 10% unchanged
        is_selected_1 = is_selected & (torch.rand_like(x, dtype=torch.float) < 0.9)
        is_selected_2 = is_selected_1 & (torch.rand_like(x, dtype=torch.float) < 1 / 9)
        x[is_selected_1] = self.mask_token_id

        # Based on the assumption that the id of the first
        # non-special token is self.num_special_tokens
        x[is_selected_2] = torch.randint(self.num_special_tokens,
                                         self.vocab_size,
                                         size=(is_selected_2.sum(),),
                                         device=x.device)

        # ignore labels of non-selected elements
        labels[~is_selected] = -100
        return x, labels


class PerceiverMLM(nn.Module):
    def __init__(self,
                 encoder: PerceiverEncoder,
                 decoder: PerceiverDecoder,
                 masking: TextMasking):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking = masking

    def forward(self, x_input, pad_mask=None, masking=True):
        _, l = x_input.shape

        if masking:
            x_masked, x_labels = self.masking(x_input, pad_mask)
        else:
            x_masked = x_input
            x_labels = None

        x_latent = self.encoder(x_masked, pad_mask)
        x_logits = self.decoder(x_latent)[:, :l, :]

        return x_logits, x_labels


class PerceiverIO(Sequential):
    def __init__(self,
                 encoder: PerceiverEncoder,
                 decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)
