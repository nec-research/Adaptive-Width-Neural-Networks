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
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from mlwiz.static import TRAINING, VALIDATION, TEST
from mlwiz.training.callback.plotter import Plotter
from mlwiz.training.event.state import State

from model import AWN


class WidthPlotter(Plotter):

    def on_epoch_end(self, state: State):
        super().on_epoch_end(state)

        variational_Ws = state.model.variational_Ws

        # Plot one graph with the different widths
        qW_probs = [w.compute_probability_vector() for w in variational_Ws]
        widths = [p.shape[0] for p in qW_probs]

        widths_dict = {f"width_{i+1}": widths[i] for i in range(len(widths))}
        self.writer.add_scalars(f"Width", widths_dict, state.epoch)

        if "model_widths" not in self.stored_metrics:
            self.stored_metrics["model_widths"] = [widths]
        else:
            self.stored_metrics["model_widths"].append(widths)

        if self.store_on_disk:
            try:
                torch.save(self.stored_metrics, self.stored_metrics_path)
            except RuntimeError as e:
                print(e)


class AWNGradientPlotter(Plotter):

    def _get_layer(self, model, layer_id):
        return model.dyn_model.get_layer(layer_id)

    def on_backward(self, state: State):
        epoch = state.epoch
        model = state.model

        num_hidden_layers = model.num_hidden_layers
        layer_wise_weight_gradients = []
        layer_wise_bias_gradients = []

        # extract gradient of weights and biases
        for l in range(num_hidden_layers + 1):
            layer = self._get_layer(model, l)

            nonzero_wg = layer.weight.grad.detach().cpu().reshape(-1)
            nonzero_wg = nonzero_wg[nonzero_wg != 0.0]
            layer_wise_weight_gradients.append(nonzero_wg)

            nonzero_bg = layer.bias.grad.detach().cpu().reshape(-1)
            nonzero_bg = nonzero_bg[nonzero_bg != 0.0]
            layer_wise_bias_gradients.append(nonzero_bg)

        if "gradients" not in self.stored_metrics:
            self.stored_metrics["gradients"] = []
            assert epoch == 0

        if len(self.stored_metrics["gradients"]) <= epoch:
            # print('epoch ', epoch)

            # new epoch, append new empty list to be filled with gradients
            # from minibatches (see "else" branch)
            self.stored_metrics["gradients"].append([])
            self.stored_metrics["gradients"][epoch] = {
                "layer_wise_weight_gradients": layer_wise_weight_gradients,
                "layer_wise_bias_gradients": layer_wise_bias_gradients,
            }
        else:
            # concatenate batch of gradients with previous ones
            lwg = self.stored_metrics["gradients"][epoch][
                "layer_wise_weight_gradients"
            ]
            lbg = self.stored_metrics["gradients"][epoch][
                "layer_wise_bias_gradients"
            ]

            # concatenate, for each layer, the gradients for this minibatch
            new_layer_wise_weight_gradients = []
            new_layer_wise_bias_gradients = []
            for l in range(num_hidden_layers + 1):
                # print(lwg[l].shape, layer_wise_weight_gradients[l].shape)

                lwg_new = torch.cat(
                    (lwg[l], layer_wise_weight_gradients[l]), dim=0
                )
                new_layer_wise_weight_gradients.append(lwg_new)
                lbg_new = torch.cat(
                    (lbg[l], layer_wise_bias_gradients[l]), dim=0
                )
                new_layer_wise_bias_gradients.append(lbg_new)

            self.stored_metrics["gradients"][epoch] = {
                "layer_wise_weight_gradients": new_layer_wise_weight_gradients,
                "layer_wise_bias_gradients": new_layer_wise_bias_gradients,
            }

    def on_training_epoch_end(self, state: State):
        epoch = state.epoch
        model = state.model
        num_hidden_layers = model.num_hidden_layers

        # compute gradient stats for this epoch
        gradient_values = self.stored_metrics["gradients"][epoch]

        if "layer_wise_weight_gradients" in gradient_values:
            lwg = gradient_values["layer_wise_weight_gradients"]
            lbg = gradient_values["layer_wise_bias_gradients"]

            if epoch % 100 == 0:
                plt.figure()
                for l in range(num_hidden_layers + 1):
                    sns.kdeplot(lwg[l], label=f"W Matrix {l}", alpha=0.5)
                plt.title(f"Weight Gradient Distribution Epoch {epoch+1}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    Path(self.exp_path, f"w_grad_kde_epoch_{epoch}.pdf")
                )
                plt.close()

                plt.figure()
                for l in range(num_hidden_layers + 1):
                    sns.kdeplot(lbg[l], label=f"W Matrix {l + 1}", alpha=0.5)
                plt.title(f"Bias Gradient Distribution Epoch {epoch+1}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    Path(self.exp_path, f"b_grad_kde_epoch_{epoch}.pdf")
                )
                plt.close()

    def on_epoch_end(self, state: State):
        epoch = state.epoch
        model = state.model
        num_hidden_layers = model.num_hidden_layers

        # compute gradient stats for this epoch
        gradient_values = self.stored_metrics["gradients"][epoch]

        if "layer_wise_weight_gradients" in gradient_values:
            lwg = gradient_values["layer_wise_weight_gradients"]
            lbg = gradient_values["layer_wise_bias_gradients"]

            weight_stats = [
                (lwg[l].mean().item(), lwg[l].std().item())
                for l in range(num_hidden_layers + 1)
            ]
            bias_stats = [
                (lbg[l].mean().item(), lbg[l].std().item())
                for l in range(num_hidden_layers + 1)
            ]

            self.stored_metrics["gradients"][epoch] = {
                "layer_wise_weight_gradients_stats": weight_stats,
                "layer_wise_bias_gradients_stats": bias_stats,
            }
            # self.stored_metrics['gradients'][epoch]['layer_wise_weight_gradients_stats'] = weight_stats
            # self.stored_metrics['gradients'][epoch]['layer_wise_bias_gradients_stats'] = bias_stats

        super().on_epoch_end(state)


class MLPGradientPlotter(AWNGradientPlotter):

    def _get_layer(self, model, layer_id):
        return model.get_layer(layer_id)


class MiniBatchPlotter(Plotter):

    def __init__(
        self, exp_path: str, store_on_disk: bool = False, **kwargs: dict
    ):
        super().__init__(exp_path, store_on_disk, **kwargs)
        self.training_batch_counter = 0.0
        self.validation_batch_counter = 0.0
        self.test_batch_counter = 0.0

    def on_epoch_end(self, state: State):
        pass

    def on_training_batch_end(self, state: State):
        assert state.set == TRAINING
        set = TRAINING

        for k, v in state.batch_loss.items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = k
            loss_scalars[f"{set}"] = v.detach().cpu()
            self.writer.add_scalars(
                loss_name, loss_scalars, self.training_batch_counter
            )

        if state.batch_score is not None:
            for k, v in state.batch_score.items():
                loss_scalars = {}
                # Remove training/validation/test prefix (coupling with Engine)
                loss_name = k
                loss_scalars[f"{set}"] = v.detach().cpu()
                self.writer.add_scalars(
                    loss_name, loss_scalars, self.training_batch_counter
                )

        if isinstance(state.model, AWN):
            variational_Ws = state.model.variational_Ws

            # Plot one graph with the different widths
            qW_probs = [w.compute_probability_vector() for w in variational_Ws]
            widths = [p.shape[0] for p in qW_probs]

            widths_dict = {
                f"width_{i+1}": widths[i] for i in range(len(widths))
            }
            self.writer.add_scalars(
                f"Width", widths_dict, self.training_batch_counter
            )

            if "model_widths" not in self.stored_metrics:
                self.stored_metrics["model_widths"] = [widths]
            else:
                self.stored_metrics["model_widths"].append(widths)

            if self.store_on_disk:
                try:
                    torch.save(self.stored_metrics, self.stored_metrics_path)
                except RuntimeError as e:
                    print(e)

        self.training_batch_counter += 1

    def on_eval_batch_end(self, state: State):
        if state.set == TRAINING:
            set = TRAINING
            bc = self.training_batch_counter
        if state.set == VALIDATION:
            set = VALIDATION
            bc = self.validation_batch_counter
        elif state.set == TEST:
            set = TEST
            bc = self.test_batch_counter

        for k, v in state.batch_loss.items():
            loss_scalars = {}
            # Remove training/validation/test prefix (coupling with Engine)
            loss_name = k
            loss_scalars[f"{set}"] = v.detach().cpu()

            self.writer.add_scalars(loss_name, loss_scalars, bc)

        if state.batch_score is not None:
            for k, v in state.batch_score.items():
                loss_scalars = {}
                # Remove training/validation/test prefix (coupling with Engine)
                loss_name = k
                loss_scalars[f"{set}"] = v.detach().cpu()

                self.writer.add_scalars(loss_name, loss_scalars, bc)

        if state.set == VALIDATION:
            self.validation_batch_counter += 1
        elif state.set == TEST:
            self.test_batch_counter += 1

    def on_fit_end(self, state: State):
        """
        Frees resources by closing the Tensorboard writer

        Args:
            state (:class:`~training.event.state.State`):
                object holding training information
        """
        self.writer.close()
