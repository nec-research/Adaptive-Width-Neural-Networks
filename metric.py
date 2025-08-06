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
from typing import List, Tuple

import torch
from mlwiz.training.callback.metric import Metric, MulticlassClassification
from torch.nn import CrossEntropyLoss


class ELBO_Classification(Metric):
    @property
    def name(self) -> str:
        return "ELBO_Classification"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_alpha,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        # maximize log p y given x
        # we assume that each minibatch contributes as if it was the entire
        # dataset, hence we multiply by n_obs (size of training set)
        if targets.shape[1] == 1:
            targets = targets.squeeze(1)

        log_p_y = (
            -CrossEntropyLoss(reduction="mean")(output_state, targets) * n_obs
        )
        elbo = log_p_y.unsqueeze(0)

        elbo += log_p_theta
        elbo += kld_alpha

        # renormalize everything by n_obs (just scaling gradients of ELBO)
        elbo = elbo / n_obs
        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class ELBO_Regression(Metric):
    @property
    def name(self) -> str:
        return "ELBO_Regression"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_alpha,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        two = torch.Tensor([2.0]).to(targets.device)

        # maximize log p y given x
        # we assume that each minibatch contributes as if it was the entire
        # dataset, hence we multiply by n_obs (size of training set)
        log_p_y = (-torch.mean(((output_state - targets) ** 2).sum(1)) / two) * n_obs

        elbo = log_p_y.unsqueeze(0)

        elbo += log_p_theta
        elbo += kld_alpha

        # renormalize everything by n_obs (just scaling gradients of ELBO)
        elbo = elbo / n_obs
        return elbo, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        elbo = predictions
        # to maximize the elbo we need to minimize -elbo
        return -elbo.mean(0)  # sum over samples


class CELoss(Metric):
    @property
    def name(self) -> str:
        return "Cross Entropy"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_alpha,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        # maximize log p y given x
        # we assume that each minibatch contributes as if it was the entire
        # dataset, hence we multiply by n_obs (size of training set)
        if targets.shape[1] == 1:
            targets = targets.squeeze(1)

        log_p_y = CrossEntropyLoss(reduction="none")(output_state, targets)
        return log_p_y, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        log_p_y = predictions
        # to maximize the elbo we need to minimize -elbo
        return log_p_y.mean(0)  # sum over samples


class Prior_theta(Metric):
    @property
    def name(self) -> str:
        return "Prior_theta"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_alpha,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        return log_p_theta, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class Prior_gamma(Metric):
    @property
    def name(self) -> str:
        return "Prior_alpha"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_alpha,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        return kld_alpha, targets

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        # to maximize the elbo we need to minimize -elbo
        return predictions.mean(0)  # sum over samples


class TotalWidth(Metric):
    @property
    def name(self) -> str:
        return "Total Width"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_gamma,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        sum_w = torch.tensor([w.shape[0] for w in qW_probs]).float().sum()
        return (sum_w.unsqueeze(0), targets)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return predictions.mean()


class ForwardTime(Metric):
    @property
    def name(self) -> str:
        return "Forward Time"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            output_state,
            embeddings,
            log_p_theta,
            kld_gamma,
            qW_probs,
            n_obs,
            variational_Ws,
            forward_delta,
        ) = outputs

        return (torch.tensor([forward_delta]), targets)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return predictions.mean()


class MachineTranslationMulticlassClassification(MulticlassClassification):

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns output[0] as predictions and dataset targets.
        Squeezes the first dimension of output and targets to get
        single vector.

        Args:
            targets (:class:`torch.Tensor`): ground truth
            outputs (List[:class:`torch.Tensor`]): outputs of the model

        Returns:
            A tuple of tensors (predicted_values, target_values)
        """
        outputs = outputs[0]

        output_reshape = outputs.contiguous().view(-1, outputs.shape[-1])
        targets = targets.contiguous().view(-1)

        return output_reshape, targets


# class BLEU(Metric):
#
#     def get_predictions_and_targets(
#         self, targets: torch.Tensor, *outputs: List[torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns output[0] as predictions and dataset targets.
#         Squeezes the first dimension of output and targets to get
#         single vector.
#
#         Args:
#             targets (:class:`torch.Tensor`): ground truth
#             outputs (List[:class:`torch.Tensor`]): outputs of the model
#
#         Returns:
#             A tuple of tensors (predicted_values, target_values)
#         """
#         outputs = outputs[0]
#
#         output_reshape = outputs.contiguous().view(-1, outputs.shape[-1])
#
#         targets = targets.contiguous().view(-1)
#
#         return output_reshape, targets
