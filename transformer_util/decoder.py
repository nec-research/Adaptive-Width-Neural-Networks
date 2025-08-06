"""
Adapted from
@author : Hyunwoong
@homepage : https://github.com/hyunwoongko/transformer/
"""

from typing import Optional

from torch import nn, Tensor
from torch.nn import Module

from dynamic_architecture import DynamicMLP
from transformer_util.attention import MultiHeadAttention
from transformer_util.embedding import TransformerEmbedding
from transformer_util.layer_norm import LayerNorm
from transformer_util.pos_fnn import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        ffn_hidden,
        n_head,
        drop_prob,
        dynamic=False,
        config=None,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, n_head=n_head
        )
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(
            d_model=d_model, n_head=n_head
        )
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.dynamic = dynamic
        if not dynamic:
            self.ffn = PositionwiseFeedForward(
                d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
            )
        else:
            num_hidden_layers = config["num_hidden_layers"]
            config["num_hidden_layers"] = 1
            self.ffn = DynamicMLP(
                dim_input_features=d_model, dim_target=d_model, config=config
            )
            config["num_hidden_layers"] = num_hidden_layers

        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask, qW_probs=None):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x

        if not self.dynamic:
            x = self.ffn(x)
        else:
            x, _, _ = self.ffn(
                x, [qW_probs]
            )  # NOTE: hardcode single ffn layer for transformer

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        if not self.dynamic:
            return x
        else:
            return x, x, None

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        return self.ffn.get_layer(layer_id)

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        return self.ffn.set_layer(layer_id, layer)

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
        neurons_probs: Optional[Tensor] = None,
    ):
        """
        Changes input or output dimension of the layer.
        :param layer_id: id of the layer to be modified
        :param num_neurons: the new number of neurons for the
            input or output dimension
        :param change_output: whether to change the output dimension.
            If false, it will change the input dimension
        :param neurons_probs: optional tensor of neuron probabilities
            for weight initialization
        :return:
        """
        self.ffn.change_shape(
            layer_id, num_neurons, change_output, neurons_probs
        )


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        dynamic=False,
        config=None,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                    dynamic=dynamic,
                    config=config,
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(d_model, dec_voc_size)
        self.dynamic = dynamic

    def forward(self, trg, enc_src, trg_mask, src_mask, qW_probs=None):
        trg = self.emb(trg)

        for layer_id, layer in enumerate(self.layers):
            if not self.dynamic:
                trg = layer(trg, enc_src, trg_mask, src_mask)
            else:
                trg, _, _ = layer(
                    trg,
                    enc_src,
                    trg_mask,
                    src_mask,
                    qW_probs=qW_probs[layer_id],
                )

        # pass to LM head
        output = self.linear(trg)
        return output

    def to(self, device):
        super().to(device)
        self.emb.to(device)

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        return self.layers[layer_id]

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        self.layers[layer_id] = layer

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
        neurons_probs: Optional[Tensor] = None,
    ):
        """
        Changes input or output dimension of the layer.
        :param layer_id: id of the layer to be modified
        :param num_neurons: the new number of neurons for the
            input or output dimension
        :param change_output: whether to change the output dimension.
            If false, it will change the input dimension
        :param neurons_probs: optional tensor of neuron probabilities
            for weight initialization
        :return:
        """
        # this is just a check that the DynamicTransformer is calling this method
        # only when change_output is true, which is a patch (see DynamicTransformer)
        assert change_output

        # assume the ffn of each layer has only 1 hidden layer
        self.layers[layer_id].change_shape(
            0, num_neurons, True, None
        )  # unused anyhow

        self.layers[layer_id].change_shape(
            1, num_neurons, False, None
        )  # unused anyhow
