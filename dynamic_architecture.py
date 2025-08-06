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
import math
from typing import Callable, Tuple, Optional, List, Union

import torch
from mlwiz.model.interface import ModelInterface
from mlwiz.util import s2c
from torch import Tensor, nn
from torch.nn import Module, Linear, Parameter, ModuleList, BatchNorm1d, \
    Sequential, ReLU
from torch.nn.functional import avg_pool2d, relu
from torch.nn.init import kaiming_normal_, normal_
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from resnet_util import BasicBlock, _weights_init


class ElementwiseProductWithScaledGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, probs):
        # Save input tensors and scaling factor for backward
        ctx.save_for_backward(h, probs)

        # Perform the element-wise product
        return h * probs

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and scaling factor
        h, probs = ctx.saved_tensors

        # Calculate gradients for each input, scaled by factor c
        grad_input_h = grad_output
        grad_input_probs = grad_output * h

        return grad_input_h, grad_input_probs


class DynamicArchitecture(ModelInterface):
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

        # Set device to cpu to avoid issues at initialization
        self.device = "cpu"

    def to(self, device):
        super().to(device)
        self.device = device

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        raise NotImplementedError(
            "You should subclass " "and implement this method"
        )

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        raise NotImplementedError(
            "You should subclass " "and implement this method"
        )

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
    ):
        """
        Changes input or output dimension of the layer.
        :param layer_id: id of the layer to be modified
        :param num_neurons: the new number of neurons for the
            input or output dimension
        :param change_output: whether to change the output dimension.
            If false, it will change the input dimension
        :return:
        """
        raise NotImplementedError(
            "You should subclass " "and implement this method"
        )

    def forward(
        self, data: Batch, qW_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        raise NotImplementedError(
            "You should subclass " "and implement this method"
        )


class DynamicMLP(DynamicArchitecture):
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

        self.num_layers = config[
            "num_hidden_layers"
        ]  # number of hidden layers
        self.num_neurons = config["initial_num_neurons"]

        a = config.get("activation", "torch.nn.functional.relu6")
        self.activation = s2c(a)

        # used to test the weight initialization scheme
        self.init_type = self.config.get("init_type", None)

        # whether to keep the initial width of the layers fixed
        self.disable_adaptation = config.get("disable_adaptation", False)

        self.apply_ntk_reparam = config.get("apply_ntk_reparam", False)
        self.apply_experimental_rescaling = config.get(
            "apply_experimental_rescaling", False
        )

        ao = config.get("activation_outer", "torch.nn.functional.leaky_relu")
        if ao is None:
            self.activation_outer = None
        else:
            self.activation_outer = s2c(ao)

        self.layers = ModuleList()

        has_bias = config.get("has_bias", True)

        # add input layer
        self.layers.append(
            Linear(dim_input_features, self.num_neurons, bias=has_bias)
        )

        # add hidden layers
        for i in range(self.num_layers - 1):  # there is already 1 hidden layer
            self.layers.append(
                Linear(self.num_neurons, self.num_neurons, bias=has_bias)
            )

        # add output layer
        self.layers.append(
            Linear(self.num_neurons, self.dim_target, bias=has_bias)
        )

        # initialize using He initialization for relu like activations
        if "leaky_relu" in a:
            self.nonlin = "leaky_relu"
        else:
            self.nonlin = "relu"

        qW_probs = config["initial_probabilities"]
        # config.pop("initial_probabilities")

        self.mode = "fan_in"

        for i, l in enumerate(self.layers):
            # First weight matrix has to be initialized as usual
            if i > 0 and (
                self.init_type is not None
                and self.init_type in ["gaussian", "uniform"]
            ):
                out_size = l.weight.shape[0]
                in_size = l.weight.shape[1]

                self._weight_init(
                    l.weight,
                    in_size,
                    out_size,
                    qW_probs[i - 1],
                    nonlinearity=self.nonlin,
                    init_type=self.init_type,
                )
            else:
                kaiming_normal_(
                    l.weight, nonlinearity=self.nonlin, mode=self.mode
                )

    def _weight_init(
        self,
        w: Parameter,
        in_size: int,
        out_size: int,
        neurons_probs: Tensor,
        nonlinearity: str,
        init_type: str,
    ):
        """
        Replaces usual weight initialization function of torch because we might
        want to add a single dimension to a weight matrix, and we need to know
        the size of the entire weight matrix to initialize "properly"
        :param w: parameter with linear matrix to be initialized
        :param in_size: size of the input dimension
        :param out_size: size of the output dimension
        :param neurons_probs: probabilities for each neuron in the layer
        :param nonlinearity: type of nonlinearity
        :param init_type: "fan_in" or "fan_out"
        :return:
        """
        assert init_type in ["gaussian", "uniform"]

        squared_sum_probs = (neurons_probs**2).sum().detach()
        if init_type == "gaussian":
            std = torch.sqrt( 2.0 / squared_sum_probs)
            with torch.no_grad():
                return w.normal_(0, std)
        # # TODO testing new init
        # D = torch.tensor([neurons_probs.shape[0]]).to(neurons_probs.device)
        # squared_sum_probs = (D*(neurons_probs**2)).detach()
        # squared_sum_probs = squared_sum_probs.unsqueeze(0).repeat(w.data.shape[0], 1)
        # if init_type == "gaussian":
        #     std = torch.sqrt( 2.0 / squared_sum_probs)
        #     with torch.no_grad():
        #         return torch.normal(torch.zeros(std.shape), std, out=w.data)
        else:
            b = math.sqrt(6.0 / float(squared_sum_probs))
            a = -b
            # uniform
            with torch.no_grad():
                return torch.nn.init.uniform_(w, a=a, b=b)

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        assert layer_id < len(self.layers)
        return self.layers[layer_id]

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        assert layer_id < len(self.layers)
        self.layers[layer_id] = layer

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
    ):
        # print(layer_id, neurons_probs)

        # layer_id starts from 0, the first mapping of the input features
        layer = self.layers[layer_id]
        has_bias = layer.bias is not None

        # store current device
        current_device = layer.weight.device

        out_size, in_size = layer.weight.shape[0], layer.weight.shape[1]
        dim_to_change = 0 if change_output else 1
        size_to_check = out_size if change_output else in_size

        changed = 0

        if (
            size_to_check < num_neurons
        ):  # we have to reduce the output size of layer layer_id and the input size of layer_id+1

            changed = 1
            if change_output:
                to_add = num_neurons - out_size
            else:
                to_add = num_neurons - in_size

            # randomly initialize the new neurons according to a standard Gaussian distribution
            if change_output:
                # add new rows to weight matrix
                new_neurons_W = torch.empty(
                    [to_add, in_size], device=current_device
                )
            else:
                # add new columns to weight matrix
                new_neurons_W = torch.empty(
                    [out_size, to_add], device=current_device
                )

            normal_(new_neurons_W)

            # concatenate the old weights with the new weights
            new_weight_W = torch.cat(
                [layer.weight, new_neurons_W], dim=dim_to_change
            )

            if has_bias and change_output:
                new_neurons_bias = torch.empty([to_add], device=current_device)

                normal_(new_neurons_bias)

                # concatenate the old weights with the new weights
                new_weight_bias = torch.cat(
                    [layer.bias, new_neurons_bias], dim=dim_to_change
                )

            # reset weight and grad variables to new size
            if change_output:
                new_layer = Linear(in_size, out_size + to_add, bias=has_bias)
            else:
                new_layer = Linear(in_size + to_add, out_size, bias=has_bias)

            # set the weight data to new values
            new_layer.weight = Parameter(new_weight_W, requires_grad=True)

            if has_bias and change_output:
                new_layer.bias = Parameter(new_weight_bias, requires_grad=True)
            elif (
                has_bias
            ):  # no need to change bias when modifying input dimension
                new_layer.bias = Parameter(layer.bias, requires_grad=True)

            self.set_layer(layer_id, new_layer)

        elif size_to_check > num_neurons:
            changed = 1
            new_weight_W = (
                layer.weight[:num_neurons, :]
                if change_output
                else layer.weight[:, :num_neurons]
            )

            if has_bias:
                new_weight_bias = (
                    layer.bias[:num_neurons] if change_output else layer.bias
                )

            # reset weight and grad variables to new size
            if change_output:
                new_layer = Linear(in_size, num_neurons, bias=has_bias)
            else:
                new_layer = Linear(num_neurons, out_size, bias=has_bias)

            # set the weight data to new values
            new_layer.weight = Parameter(new_weight_W, requires_grad=True)
            if has_bias:
                new_layer.bias = Parameter(new_weight_bias, requires_grad=True)

            self.set_layer(layer_id, new_layer)
        return changed

    def random_truncation(
        self,
        layer_id: int,
        num_neurons_to_remove: int,
        change_output: bool,
    ):
        # layer_id starts from 0, the first mapping of the input features
        layer = self.get_layer(layer_id)
        has_bias = layer.bias is not None

        out_size, in_size = layer.weight.shape[0], layer.weight.shape[1]
        size_to_check = out_size if change_output else in_size

        if num_neurons_to_remove == size_to_check:
            return

        assert size_to_check > num_neurons_to_remove

        # Randomly Select neurons to remove (for study purposes only!)
        indices_to_remove = torch.randperm(size_to_check)[:num_neurons_to_remove]
        neurons_to_keep = torch.ones(size_to_check, dtype=torch.bool)
        neurons_to_keep[indices_to_remove] = False

        assert len(indices_to_remove) == num_neurons_to_remove

        new_weight_W = (
            layer.weight[neurons_to_keep, :]
            if change_output
            else layer.weight[:, neurons_to_keep]
        )

        if has_bias:
            new_weight_bias = (
                layer.bias[neurons_to_keep] if change_output else layer.bias
            )

        new_size = int(neurons_to_keep.sum())

        # reset weight and grad variables to new size
        if change_output:
            new_layer = Linear(in_size, new_size, bias=has_bias)
        else:
            new_layer = Linear(new_size, out_size, bias=has_bias)

        # set the weight data to new values
        new_layer.weight = Parameter(new_weight_W, requires_grad=True)
        if has_bias:
            new_layer.bias = Parameter(new_weight_bias, requires_grad=True)

        self.set_layer(layer_id, new_layer)


    def index_truncation(
        self,
        layer_id: int,
        indexes_to_remove: torch.Tensor,
        change_output: bool,
    ):
        # layer_id starts from 0, the first mapping of the input features
        layer = self.get_layer(layer_id)
        has_bias = layer.bias is not None

        out_size, in_size = layer.weight.shape[0], layer.weight.shape[1]
        size_to_check = out_size if change_output else in_size

        if len(indexes_to_remove) == size_to_check:
            return

        assert size_to_check > len(indexes_to_remove)

        # Randomly Select neurons to remove (for study purposes only!)
        neurons_to_keep = torch.ones(size_to_check, dtype=torch.bool)
        neurons_to_keep[indexes_to_remove] = False

        new_weight_W = (
            layer.weight[neurons_to_keep, :]
            if change_output
            else layer.weight[:, neurons_to_keep]
        )

        if has_bias:
            new_weight_bias = (
                layer.bias[neurons_to_keep] if change_output else layer.bias
            )

        new_size = int(neurons_to_keep.sum())

        # reset weight and grad variables to new size
        if change_output:
            new_layer = Linear(in_size, new_size, bias=has_bias)
        else:
            new_layer = Linear(new_size, out_size, bias=has_bias)

        # set the weight data to new values
        new_layer.weight = Parameter(new_weight_W, requires_grad=True)
        if has_bias:
            new_layer.bias = Parameter(new_weight_bias, requires_grad=True)

        self.set_layer(layer_id, new_layer)


    def forward(
        self,
        data: Batch,
        qW_probs: List[torch.Tensor],
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        pre_activations = []
        activations = []
        post_filter = []

        # h = data.to(torch.get_default_dtype())
        h = data

        for l in range(self.num_layers):
            layer = self.get_layer(layer_id=l)

            neurons_probs = qW_probs[l].unsqueeze(0)  # 1 x current width

            if len(h.shape) == 3:  # TRANSFOMER CASE
                neurons_probs = neurons_probs.unsqueeze(
                    0
                )  # 1 x 1 x current width

            pre_h = layer(h)

            if return_activations:
                pre_activations.append(pre_h)

            # apply weight distribution after non-linear activation
            h = self.activation(pre_h)

            if return_activations:
                activations.append(h)

            if not self.disable_adaptation:
                # TODO ONLY FOR DEBUG/RESEARCH PURPOSES
                # MIGHT CAUSE BUGS.
                # Used when I am testing pruning, make sure neurons_probs are
                # adequately rescaled
                if h.shape[1] < neurons_probs.shape[1]:
                    assert not self.training
                    h_dim = h.shape[1]
                    neurons_probs = neurons_probs[:, :h_dim]

                if self.apply_ntk_reparam:
                    rescaling = torch.tensor(
                        [len(neurons_probs)],
                        dtype=torch.float32,
                        device=h.device,
                    )
                    h = h / torch.sqrt(rescaling)

                # TODO EXPERIMENTAL
                if self.apply_experimental_rescaling:
                    h = ElementwiseProductWithScaledGrad.apply(
                        h, neurons_probs
                    )
                else:
                    h = h * neurons_probs

                if return_activations:
                    post_filter.append(h)

                if self.activation_outer is not None:
                    h = self.activation_outer(h)

        out_layer = self.get_layer(layer_id=l + 1)
        o = out_layer(h)

        if return_activations:
            return o, h, (pre_activations, activations, post_filter)
        else:
            return o, h, None

    def __len__(self):
        return len(self.layers)


class DynamicResNet20(DynamicArchitecture):

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

        self.apply(_weights_init)

        self.fc = DynamicMLP(64, dim_target, config)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        return self.fc.get_layer(layer_id)

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        return self.fc.set_layer(layer_id, layer)

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
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
        self.fc.change_shape(layer_id, num_neurons, change_output)

    def forward(
        self,
        x: Batch,
        qW_probs: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out, h, extra = self.fc(out, qW_probs, return_activations)
        return out, h, extra

    def __len__(self):
        return len(self.fc)


class DynamicRNN(DynamicArchitecture):
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

        # Learnable parameters

        # SEE PATCH in config file
        config["num_hidden_layers"] = config["num_hidden_layers"]//2

        self.input_to_hidden = DynamicMLP(dim_input, self.hidden_dim, config)
        self.hidden_to_hidden = DynamicMLP(self.hidden_dim, self.hidden_dim, config)
        config["num_hidden_layers"] = config["num_hidden_layers"]*2

        self.num_layers = config["num_hidden_layers"]

        self.activation = nn.Tanh()  # Activation function for hidden state

        self.out_layer = Linear(self.hidden_dim, dim_target)

    def forward(self, data: torch.Tensor,
                qW_probs: torch.Tensor,
                return_activations=False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        batch_size, sequence_length, _ = data.size()
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=data.device)

        for t in range(sequence_length):
            input_t = data[:, t, :]  # Get the t-th timestep input [batch_size, input_dim]
            hidden_state = self.activation(
                self.input_to_hidden(input_t, qW_probs[:self.num_layers//2])[0] + self.hidden_to_hidden(hidden_state, qW_probs[self.num_layers//2:])[0]
            )  # Update hidden state

        o = self.out_layer(hidden_state)

        return o, hidden_state, None

    def __len__(self):
        return self.num_layers

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        if layer_id < self.num_layers // 2:
            return self.input_to_hidden.get_layer(layer_id)
        else:
            return self.hidden_to_hidden.get_layer(layer_id % 2)

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        if layer_id < self.num_layers // 2:
            return self.input_to_hidden.set_layer(layer_id, layer)
        else:
            return self.v.set_layer(layer_id % 2, layer)

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
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
        # PATCH
        half = self.num_layers // 2
        if layer_id < (self.num_layers // 2) and change_output:
            self.input_to_hidden.change_shape(layer_id, num_neurons, True)
            self.input_to_hidden.change_shape(layer_id+1, num_neurons, False)
        elif layer_id < self.num_layers and change_output:
            self.hidden_to_hidden.change_shape(layer_id%half, num_neurons, True)
            self.hidden_to_hidden.change_shape((layer_id%half)+1, num_neurons, False)
        else:
            pass


class ModGINConv(GINConv):
    def __init__(
        self,
        nn: Callable,
        embedding_dim,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.bn = BatchNorm1d(embedding_dim)

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, qW_probs, size=None) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        # this is the output of a DynamicMLP now
        out, _, _ = self.nn(out, qW_probs)
        return relu(self.bn(out))


class DynamicDGN(DynamicArchitecture):
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
        self.embedding_dim = config["embedding_dim"]

        pool = config.get("global_pooling", None)
        if pool is None:
            self.global_pool = None
        elif pool == "sum":
            self.global_pool = global_add_pool
        elif pool == "mean":
            self.global_pool = global_mean_pool
        else:
            raise NotImplementedError("Pooling not implemented")

        self.num_neurons = config["initial_num_neurons"]

        self.layers = ModuleList()

        self.layers = ModuleList()
        self.out_layers = ModuleList()

        has_bias = config.get("has_bias", True)

        # PATCH: change config to inform MLP that only 1 hidden layer
        # has to be used
        hl = config["num_hidden_layers"]
        config["num_hidden_layers"] = 1
        # add input layer
        self.layers.append(
            DynamicMLP(dim_input_features, self.embedding_dim, config)
        )
        config["num_hidden_layers"] = hl
        # add output layer: Linear layer on concatenation of node/graph embeddings
        self.out_layers.append(
            Linear(self.embedding_dim, self.dim_target, bias=has_bias)
        )

        # add hidden layers
        for i in range(self.num_hidden_layers - 1):
            # PATCH: change config to inform MLP that only 1 hidden layer
            # has to be used
            hl = config["num_hidden_layers"]
            config["num_hidden_layers"] = 1
            self.layers.append(
                ModGINConv(
                    DynamicMLP(self.embedding_dim, self.embedding_dim, config),
                    self.embedding_dim,
                    train_eps=True,
                )
            )
            config["num_hidden_layers"] = hl

            # add output layer: Linear layer on concatenation of node/graph embeddings
            self.out_layers.append(
                Linear(self.embedding_dim, self.dim_target, bias=has_bias)
            )

    def get_layer(self, layer_id: int) -> Module:
        """
        Retrieves the layer to be modified.
        :param layer_id: id of the layer to retrieve
        :return: a Module object holding the layer
        """
        assert layer_id < len(self)
        if layer_id == 0:
            return self.layers[layer_id]
        else:
            return self.layers[layer_id].nn

    def set_layer(self, layer_id: int, layer):
        """
        Replace the current layer with the modified one
        :param layer_id: id of the layer to replace
        :param layer: a Module object holding the new layer
        :return:
        """
        assert layer_id < len(self)
        if layer_id == 0:
            self.layers[layer_id] = layer
        else:
            self.layers[layer_id].nn = layer

    def change_shape(
        self,
        layer_id: int,
        num_neurons: int,
        change_output: bool,
    ):

        # PATCH TO WORK WITH AWN, which assumes a too simple structure
        if change_output:
            assert layer_id < len(self)
            if layer_id == 0:
                # assume the ffn of each layer has only 1 hidden layer
                self.layers[layer_id].change_shape(0, num_neurons, True)

                self.layers[layer_id].change_shape(1, num_neurons, False)
            else:
                # assume the ffn of each layer has only 1 hidden layer
                self.layers[layer_id].nn.change_shape(0, num_neurons, True)

                self.layers[layer_id].nn.change_shape(1, num_neurons, False)

        else:
            # we are already taking care of l+1 inside the transformer
            # since the layer index here is used to refer to either the encoder
            # or decoder layer, and not the internal FFN
            pass

    def forward(
        self,
        data: Batch,
        qW_probs: List[torch.Tensor],
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0.0

        # conv layers
        for l in range(self.num_hidden_layers):
            if l == 0:
                h, _, _ = self.layers[l](x, [qW_probs[l]])  # first MLP
            else:
                h = self.layers[l](h, edge_index, [qW_probs[l]])

            if self.global_pool is not None:
                h_pool = self.global_pool(h, batch)
                out += self.out_layers[l](h_pool)
            else:
                out += self.out_layers[l](h)

        return out, h, None

    def __len__(self):
        return len(self.layers)
