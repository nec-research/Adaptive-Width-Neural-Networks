"""
       Adaptive Width Neural Networks
	  
  File:     distribution.py, dynamic_transformer.py, experiment.py, model.py, metric.py, plotter.py, scheduler.py, transform.py
  Authors:  Federico Errica (federico.errica@neclab.eu)
            Henrik Christiansen (henrik.christiansen@neclab.eu)
	    Viktor Zaverkin (viktor.zaverkin@neclab.eu)
            Mathias Niepert (mathias.niepert@ki.uni-stuttgart.de)
            Francesco Alesiani (francesco.alesiani@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.  

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
       PROPRIETARY INFORMATION ---  

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.  

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.  

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from dynamic_architecture import DynamicArchitecture
from transformer_util.decoder import Decoder
from transformer_util.encoder import Encoder


# https://github.com/hyunwoongko/transformer/
class DynamicTransformer(DynamicArchitecture):

    def __init__(
        self,
        dim_input_features: int,
        dim_target: int,
        config: dict,
    ):
        super().__init__(dim_input_features, dim_target, config)
        self.max_len = dim_input_features
        self.d_model = config["embedding_dim"]

        # assume same tokenizer for encoding and decoding
        self.enc_vocab_size = dim_target
        self.dec_vocab_size = dim_target

        self.n_head = config["num_attention_heads"]

        self.ffn_hidden = config.get("num_hidden_neurons", None)

        self.n_layers = config["num_enc_dec_layers"]

        self.drop_prob = config["dropout"]

        self.encoder = Encoder(
            d_model=self.d_model,
            n_head=self.n_head,
            max_len=self.max_len,
            ffn_hidden=self.ffn_hidden,
            enc_voc_size=self.enc_vocab_size,
            drop_prob=self.drop_prob,
            n_layers=self.n_layers,
            dynamic=True,
            config=config,
        )

        self.decoder = Decoder(
            d_model=self.d_model,
            n_head=self.n_head,
            max_len=self.max_len,
            ffn_hidden=self.ffn_hidden,
            dec_voc_size=self.dec_vocab_size,
            drop_prob=self.drop_prob,
            n_layers=self.n_layers,
            dynamic=True,
            config=config,
        )

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        assert layer_id < self.n_layers * 2
        if layer_id >= self.n_layers:
            # print(f'Retrieving decoder layer {layer_id} == {layer_id % self.n_layers}')
            return self.decoder.get_layer(layer_id % self.n_layers)
        else:
            # print(f'Retrieving layer {layer_id} == encoder layer {layer_id % self.n_layers}')
            return self.encoder.get_layer(layer_id % self.n_layers)

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        assert layer_id < self.n_layers * 2
        if layer_id >= self.n_layers:
            return self.decoder.set_layer(layer_id % self.n_layers, layer)
        else:
            return self.encoder.set_layer(layer_id % self.n_layers, layer)

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
        # PATCH TO WORK WITH AWN, which assumes a simple structure
        if change_output:
            assert layer_id < self.n_layers * 2
            if layer_id >= self.n_layers:
                self.decoder.change_shape(
                    layer_id % self.n_layers,
                    num_neurons,
                    change_output,
                    neurons_probs,
                )
            else:
                self.encoder.change_shape(
                    layer_id % self.n_layers,
                    num_neurons,
                    change_output,
                    neurons_probs,
                )
        else:
            # we are already taking care of l+1 inside the transformer
            # since the layer index here is used to refer to either the encoder
            # or decoder layer, and not the internal FFN
            pass

    def forward(self, src_trg, qW_probs):
        assert len(qW_probs) == self.n_layers * 2

        tok, att_mask = src_trg[:, :, :, 0], src_trg[:, :, :, 1]
        src, trg = tok[:, :, 0], tok[:, :, 1]
        src_mask, trg_mask = att_mask[:, :, 0], att_mask[:, :, 1]

        # preprocess mask specific to this Transformer implementation
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask, device=tok.device)

        enc_src = self.encoder(src, src_mask, qW_probs[: self.n_layers])
        output = self.decoder(
            trg, enc_src, trg_mask, src_mask, qW_probs[self.n_layers :]
        )
        return output, enc_src, None

    def make_src_mask(self, src_mask):
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg_mask, device):
        # trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_pad_mask = trg_mask.unsqueeze(1).unsqueeze(3)
        trg_len = self.max_len  # trg.shape[1]
        trg_sub_mask = (
            torch.tril(torch.ones(trg_len, trg_len))
            .type(torch.ByteTensor)
            .to(device)
        )
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def __len__(self):
        # NOTE: THIS WILL UNIQUELY DETERMINE THE NUMBER OF ADAPTIVE LAYERS
        #  INCLUDING ENCODER AND DECODER
        return self.n_layers * 2
