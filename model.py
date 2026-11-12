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
import time
from typing import Tuple, Optional, List, Mapping, Any, Union

import torch
from mlwiz.evaluation.util import return_class_and_args
from mlwiz.model.interface import ModelInterface
from mlwiz.training.callback.optimizer import Optimizer
from mlwiz.util import s2c
from torch import nn,Tensor
from torch.nn import (
    ModuleList,
    Linear,
    Sequential,
    init,
    ReLU,
    BatchNorm1d,
    LeakyReLU,
)
from torch.nn.functional import avg_pool2d
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GINConv

from distribution import TruncatedDistribution
from dynamic_architecture import DynamicArchitecture, DynamicMLP
from dynamic_optimizers import update_optimizer
from resnet_util import BasicBlock, _weights_init
from transformer_util.decoder import Decoder
from transformer_util.encoder import Encoder


class MLP(ModelInterface):
    def __init__(
        self,
        dim_input_features,
        dim_target,
        config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )

        # depth of network
        self.num_hidden_layers = config["num_hidden_layers"]

        # depth of network
        self.num_hidden_neurons = config["num_hidden_neurons"]

        # if "LogActivation" in config.get('activation'):
        #     self.activation = LogActivation()
        # else:
        self.activation = s2c(
            config.get("activation", "torch.nn.functional.leaky_relu")
        )

        self.apply_ntk_reparam = config["apply_ntk_reparam"]

        self.layers = ModuleList()

        has_bias = config.get("has_bias", True)

        # add input layer
        self.layers.append(
            Linear(dim_input_features, self.num_hidden_neurons, bias=has_bias)
        )

        # add hidden layers
        for i in range(
            self.num_hidden_layers - 1
        ):  # there is already 1 hidden layer
            self.layers.append(
                Linear(
                    self.num_hidden_neurons,
                    self.num_hidden_neurons,
                    bias=has_bias,
                )
            )

        # add output layer
        self.layers.append(
            Linear(self.num_hidden_neurons, dim_target, bias=has_bias)
        )

    def get_layer(self, layer_id: int):
        return self.layers[layer_id]

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        h = data

        for i in range(self.num_hidden_layers):
            h = self.activation(self.layers[i](h))

            if self.apply_ntk_reparam:
                h = h / torch.sqrt(torch.tensor([self.num_hidden_neurons]))

        o = self.layers[self.num_hidden_layers](h)

        return o, h


class GMNNBase(ModelInterface):

    scale: Tensor = None
    shift: Tensor = None

    def __init__(
        self,
        dim_input_features,
        dim_target,
        config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )

        # depth of network
        self.num_hidden_layers = config["num_hidden_layers"]

        # depth of network
        self.num_hidden_neurons = config["num_hidden_neurons"]

        self.layers = ModuleList()

        # add input layer
        self.layers.append(
            Linear(dim_input_features, self.num_hidden_neurons, bias=True)
        )

        self.activation = s2c(
            config.get("activation", "torch.nn.functional.relu")
        )

        # add hidden layers
        for i in range(
            self.num_hidden_layers - 1
        ):  # there is already 1 hidden layer
            self.layers.append(
                Linear(
                    self.num_hidden_neurons,
                    self.num_hidden_neurons,
                    bias=True,
                )
            )

        # add output layer
        self.layers.append(
            Linear(self.num_hidden_neurons, dim_target, bias=True)
        )

        n_radial: int = 7
        n_basis: int = 9
        emb_init: str = 'constant',
        n_species: int = 5

        self.representation = GaussianMoments(
            r_cutoff=3.0,  # NOTE: cutoff radius was fixed to 3.0
            n_radial=n_radial,
            n_basis=n_basis,
            n_species=n_species,
            emb_init=emb_init
        )

        self.scale_shift = ScaleShiftLayer(shift_params=self.shift,
                                           scale_params=self.scale)

    def get_layer(self, layer_id: int):
        return self.layers[layer_id]

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        h = self.representation(data)

        for i in range(self.num_hidden_layers):
            h = self.activation(self.layers[i](h))

        o = self.layers[self.num_hidden_layers](h)

        atomic_energies = self.scale_shift(o, data)

        #atomic_energies = o.squeeze(-1)
        total_energy = segment_sum(atomic_energies, idx_i=data.batch, dim_size=data.n_atoms.shape[0])

        return total_energy, h


class GMNNQM9Energy(GMNNBase):

    scale: Tensor = torch.tensor(
        [4.13805536, 4.13805536, 4.13805536, 4.13805536, 4.13805536])
    shift: Tensor = torch.tensor(
        [-64.97240233, -142.58802352, -103.77941241, -100.79497677,
         -92.4649521])


# adapted from https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
class ResNet20(ModelInterface):

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

    def __init__(
        self,
        dim_input_features: int,
        dim_target: int,
        config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        # Resnet20
        num_blocks = [3, 3, 3]
        self.expansion = 1

        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            dim_input_features[2],
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks[2], stride=2)

        num_hidden_neurons = config["num_hidden_neurons"]
        if num_hidden_neurons == 0:
            self.fc = nn.Linear(64, dim_target)
        else:
            self.fc = Sequential(
                nn.Linear(64, num_hidden_neurons),
                LeakyReLU(),
                nn.Linear(num_hidden_neurons, dim_target),
            )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, out


class RNN(ModelInterface):
    def __init__(
            self,
            dim_input_features: Union[int, Tuple[int]],
            dim_target: int,
            config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )
        dim_input, len_window = dim_input_features[1], dim_input_features[0]
        self.hidden_dim = config["num_hidden_neurons"]
        self.hidden_layers = config["num_hidden_layers"]

        self.linear = config["linear"]

        # Learnable parameters
        if not self.linear:
            assert self.hidden_layers in [1,2]
            if self.hidden_layers == 1:
                self.input_to_hidden = Sequential(Linear(dim_input, self.hidden_dim),
                                                     ReLU(),
                                                     Linear(self.hidden_dim, self.hidden_dim))
                self.hidden_to_hidden = Sequential(Linear(self.hidden_dim, self.hidden_dim),
                                                     ReLU(),
                                                     Linear(self.hidden_dim, self.hidden_dim))
            else:  # 2 layers
                self.input_to_hidden = Sequential(
                    Linear(dim_input, self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim))
                self.hidden_to_hidden = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                    ReLU(),
                    Linear(self.hidden_dim, self.hidden_dim))
        else:
            self.input_to_hidden = Linear(dim_input, self.hidden_dim)
            self.hidden_to_hidden = Linear(self.hidden_dim, self.hidden_dim)

        self.activation = nn.Tanh()  # Activation function for hidden state

        self.out_layer = Linear(self.hidden_dim, dim_target)

    def forward(self, data: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        batch_size, sequence_length, _ = data.size()
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=data.device)

        for t in range(sequence_length):
            input_t = data[:, t, :]  # Get the t-th timestep input [batch_size, input_dim]
            hidden_state = self.activation(
                self.input_to_hidden(input_t) + self.hidden_to_hidden(hidden_state)
            )  # Update hidden state

        o = self.out_layer(hidden_state)

        return o, hidden_state


class DGN(ModelInterface):
    def __init__(
        self,
        dim_input_features: int,
        dim_target: int,
        config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )

        self.num_hidden_layers = config["num_hidden_layers"]
        self.embedding_dim = config["num_hidden_neurons"]

        pool = config.get("global_pooling", None)
        if pool is None:
            self.global_pool = None
        elif pool == "sum":
            self.global_pool = global_add_pool
        elif pool == "mean":
            self.global_pool = global_mean_pool
        else:
            raise NotImplementedError("Pooling not implemented")

        has_bias = config.get("has_bias", True)

        aggr = config.get("aggr", None)

        self.layers = ModuleList()
        self.out_layers = ModuleList()

        # add input layer
        mlp_layer = Sequential(
            Linear(dim_input_features, self.embedding_dim),
            ReLU(),
            Linear(self.embedding_dim, self.embedding_dim),
            ReLU(),
        )
        self.layers.append(mlp_layer)
        # add output layer: Linear layer on concatenation of node/graph embeddings
        self.out_layers.append(
            Linear(self.embedding_dim, self.dim_target, bias=has_bias)
        )

        # add hidden layers
        for i in range(self.num_hidden_layers - 1):
            mlp_layer = Sequential(
                Linear(self.embedding_dim, self.embedding_dim),
                BatchNorm1d(self.embedding_dim),
                ReLU(),
                ReLU(),
                Linear(self.embedding_dim, self.embedding_dim),
                ReLU(),
                BatchNorm1d(self.embedding_dim),
                ReLU(),
            )
            conv = GINConv(mlp_layer, train_eps=True)
            if aggr is not None:
                conv.aggr = aggr

            self.layers.append(conv)

            # add output layer: Linear layer on concatenation of node/graph embeddings
            self.out_layers.append(
                Linear(self.embedding_dim, self.dim_target, bias=has_bias)
            )

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0.0

        # conv layers
        for l in range(self.num_hidden_layers):
            if l == 0:
                h = self.layers[l](x)  # first MLP
            else:
                h = self.layers[l](h, edge_index)

            # if not self.training:
            #     print("max H NOT TRAINING", l)
            #     print(torch.max(h))
            #
            # if self.training:
            #     print("max H TRAINING", l)
            #     print(torch.max(h))

            if self.global_pool is not None:
                h_pool = self.global_pool(h, batch)
                out += self.out_layers[l](h_pool)
            else:
                out += self.out_layers[l](h)

        # if not self.training:
        #     # print("X")
        #     # print(x, x.shape)
        #     # print("OUT")
        #     # print(out)
        #     print()
        #     exit(0)

        return out, h


# https://github.com/hyunwoongko/transformer/
class Transformer(ModelInterface):

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
        self.ffn_hidden = config["num_hidden_neurons"]
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
        )

        self.decoder = Decoder(
            d_model=self.d_model,
            n_head=self.n_head,
            max_len=self.max_len,
            ffn_hidden=self.ffn_hidden,
            dec_voc_size=self.dec_vocab_size,
            drop_prob=self.drop_prob,
            n_layers=self.n_layers,
        )

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, src_trg):

        tok, att_mask = src_trg[:, :, :, 0], src_trg[:, :, :, 1]
        src, trg = tok[:, :, 0], tok[:, :, 1]
        src_mask, trg_mask = att_mask[:, :, 0], att_mask[:, :, 1]

        # preprocess mask specific to this Transformer implementation
        src_mask = self.make_src_mask(src_mask)
        trg_mask = self.make_trg_mask(trg_mask, device=tok.device)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, enc_src

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


class AWN(ModelInterface):
    def __init__(
        self,
        dim_input_features: int,
        dim_target: int,
        config: dict,
    ):
        super().__init__(
            dim_input_features,
            dim_target,
            config,
        )

        config = dict(config)

        # UNUSED FOR NOW
        # # min width of each layer
        # self.min_width = int(config.get("min_width", 2))

        # depth of network
        self.num_hidden_layers = config["num_hidden_layers"]

        # to be set up later by the PyDGN engine
        self.torch_optimizer = None

        # list to be set up after initialization of parameters
        self.current_output_widths = None

        self.n_obs = config["n_observations"]

        # store the quantile we want to use
        self.quantile = config["quantile"]

        # whether to keep the initial width of the layers fixed
        self.disable_adaptation = config.get("disable_adaptation", False)

        # whether to learn the same width distribution for all layers
        self.share_width_distribution = config.get(
            "share_width_distribution", False
        )

        t_dist_cls, t_dist_args = return_class_and_args(
            config, "truncated_distribution"
        )
        assert t_dist_cls == TruncatedDistribution

        # instantiate one distribution to change the output dimensions for each
        # hidden layer
        if not self.share_width_distribution:
            self.variational_Ws = ModuleList(
                [
                    t_dist_cls(
                        truncation_quantile=self.quantile, **t_dist_args
                    )
                    for _ in range(self.num_hidden_layers)
                ]
            )
        else:
            dist = t_dist_cls(truncation_quantile=self.quantile, **t_dist_args)
            self.variational_Ws = ModuleList(
                [dist for _ in range(self.num_hidden_layers)]
            )

        for w in self.variational_Ws:
            # Necessary to initialize weights later
            dummy_device = "cpu"
            w.to(dummy_device)

        if self.disable_adaptation:
            # disable adaptation of width distributions
            for d in self.variational_Ws:
                for param in d.parameters():
                    param.requires_grad = False

        # the quantile will be the same for all layers
        initial_num_neurons = self.variational_Ws[0].quantile(self.quantile)[0]
        config["initial_num_neurons"] = int(initial_num_neurons)

        self.qW_probs = [
            w.compute_probability_vector() for w in self.variational_Ws
        ]

        config["initial_probabilities"] = [q.detach() for q in self.qW_probs]

        dyn_model_cls, dyn_model_args = return_class_and_args(
            config, "dynamic_architecture"
        )
        self.dyn_model: DynamicArchitecture = dyn_model_cls(
            dim_input_features,
            dim_target,
            {**dyn_model_args, **config},
        )

        if "initial_probabilities" in config:
            config.pop("initial_probabilities")

        # # Instantiate the variational distribution q(\theta | \ell)
        # NOTE: not needed, see comment in forward method
        # q_theta_L_cls, q_theta_L_args = s2c(config['q_theta_given_L'])
        # q_theta_L = q_theta_L_cls(q_theta_L_args)
        # self.variational_theta = q_theta_L

        # prior scale for p(theta) - we use a normal with mean 0
        self.theta_prior_scale = Parameter(
            torch.tensor([config["theta_prior_scale"]]), requires_grad=False
        )

        # prior scale for p(alpha)
        self.alpha_prior_mean = Parameter(
            torch.tensor([config["alpha_prior_mean"]]), requires_grad=False
        )

        # prior scale for p(alpha)
        _alpha_prior_scale = config.get("alpha_prior_scale", None)
        if _alpha_prior_scale is None:
            self.alpha_prior_scale = None
        else:
            self.alpha_prior_scale = Parameter(
                torch.tensor([_alpha_prior_scale]), requires_grad=False
            )

        # prior for width p(width)
        w_prior_cls, w_prior_args = return_class_and_args(
            config, "width_prior"
        )
        if w_prior_cls is not None:
            self.width_prior = w_prior_cls(**w_prior_args)
        else:
            # uninformative prior
            self.width_prior = None

        self.device = None
        self._apply_alpha_prior = True

        self.old_parameters = {k: v for k, v in self.named_parameters()}

    def to(self, device):
        """Set the device of the model."""
        super().to(device)

        self.device = device

        for w in self.variational_Ws:
            w.to(device)

        self.dyn_model.to(device)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        # first, copy only the keys related to the width distributions
        var_Ws_keys = {
            k: v for k, v in state_dict.items() if "variational_Ws" in k
        }
        super().load_state_dict(var_Ws_keys, strict=False)

        self.qW_probs = [
            w.compute_probability_vector() for w in self.variational_Ws
        ]

        # then, update the NN layers so the size matches the best checkpoint
        self.update_width()

        # finally, load the entire checkpoint
        super().load_state_dict(state_dict, strict)
        self.old_parameters = {k: v for k, v in self.named_parameters()}

    def update_width(self):
        """
        Compute the current width of each layer of the dynamic architecture.
        """
        # assert self.device is not None, "Device has not been set"

        widths = [w.compute_truncation_number() for w in self.variational_Ws]

        changed = 0
        for l, num_neurons in enumerate(widths):
            # this loop is over variational Ws because it starts by potentially
            # changing the output of the very first weight matrix
            # from the input to hidden representation).
            # This means that we should initialize neurons differently
            # from l=1 onwards.

            # change output of current layer
            changed += self.dyn_model.change_shape(
                layer_id=l,
                num_neurons=num_neurons,
                change_output=True,
            )

            # change input of next layer
            changed += self.dyn_model.change_shape(
                layer_id=l + 1,
                num_neurons=num_neurons,
                change_output=False,
            )
        
        if changed > 0:
            new_parameters = {k: v for k, v in self.named_parameters()}
            update_optimizer(
                self.torch_optimizer,
                new_parameters,
                optimized_params=self.old_parameters,
                reset_state=False,
                remove_params=True,  # remove extra parameters
                verbose=False,
            )
            self.old_parameters = new_parameters

        return widths


    def set_optimizer(self, optimizer: Optimizer):
        """
        Set the optimizer to later add the dynamically created
           layers' parameters to it.
        """
        # recover torch Optimizer object from PyDGN one
        self.torch_optimizer = optimizer.optimizer

    def get_q_w_named_parameters(self) -> List[dict]:
        return [d.get_q_named_parameters() for d in self.variational_Ws]

    def apply_alpha_prior(self, v: bool):
        self._apply_alpha_prior = v

    def forward(
        self,
        data: Union[torch.Tensor, Batch],
        skip_update: bool = False,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        forward_start = time.time()

        # skip_update used in notebooks to test pruning
        if not skip_update:
            # first, determine if layers have to be adjusted in width
            widths = self.update_width()

        # computes probability vector of variational distr. q(layer)
        self.qW_probs = [
            w.compute_probability_vector_known_width(widths[idx]) for idx, w in enumerate(self.variational_Ws)
        ]

        # gammas act as mean of gaussian first-order approximation of q_gamma
        gammas = torch.cat(
            [
                w.discretized_distribution.base_distribution.parameter
                for w in self.variational_Ws
            ],
            dim=0,
        )
        # scale of q_gamma assumed to be 1 -> logvar1 = 0
        if self.alpha_prior_scale is not None and self._apply_alpha_prior:
            # in KL(a||b), we consider b as our prior and a as the variational dist
            # Compute the KL divergence term by term for parameters
            logvar2 = torch.log(self.alpha_prior_scale)
            mu1 = gammas
            mu2 = self.alpha_prior_mean
            kld_alpha = (-logvar2 - (
                (mu1 - mu2).square() / (2.0 * self.alpha_prior_scale.square())
            )).sum(dim=0, keepdim=True)
        else:
            kld_alpha = torch.tensor([0.0], device=self.device)


        if self.theta_prior_scale is not None:
            log_p_theta_cumulative = (-sum(p.square().sum() for p in self.dyn_model.parameters())) / (2.0 * self.theta_prior_scale.square())

        # Note: since the assumption that q(theta; nu) = N(theta; nu, I), it
        # follows that log(N(nu; nu, I)) is a constant number and can be
        # avoided in the optimization process
        output, embeddings, other = self.dyn_model(
            data, self.qW_probs, return_activations=return_activations
        )

        if other is not None and return_activations:
            pre_activations, activations, post_filter = other
            embeddings = other

        forward_end = time.time()
        forward_delta = forward_end - forward_start

        return (
            output,
            embeddings,
            log_p_theta_cumulative,
            kld_alpha,
            self.qW_probs,
            self.n_obs,
            self.variational_Ws,
            forward_delta,
        )


class LayerwiseLRAWN(AWN):

    def set_optimizer(self, optimizer: Optimizer):
        """
        Set the optimizer to later add the dynamically created
           layers' parameters to it.
        """
        # recover torch Optimizer object from PyDGN one
        self.torch_optimizer = optimizer.optimizer

        # print(self.torch_optimizer.param_groups)
        for i, group in enumerate(self.torch_optimizer.param_groups):
            # print(group['lr'])
            if i == 0:
                c = 3.16  # sqrt(10)
            else:
                c = 10  # sqrt(100)
            group["lr"] = group["lr"] * c  # increase lr of last layers
