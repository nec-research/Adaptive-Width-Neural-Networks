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
from typing import Tuple

import torch
from mlwiz.evaluation.util import return_class_and_args
from torch.nn import Parameter, Module
from torch.nn.functional import softplus, relu, sigmoid


def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class ContinuousDistribution(Module):
    """
    Implements an interface for this package
    """

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.register_buffer('_one', torch.tensor(1.0, device=self.device))

    def to(self, device):
        super().to(device)
        self.device = device
        self._one.to(self.device)

    def _validate_args(self, value):
        assert isinstance(
            value, torch.Tensor
        ), f"expected torch tensor, found {type(value)}"

        # assert isinstance(value, torch.FloatTensor) or (
        #     value.dtype == torch.float32
        # ), f"expected float tensor, found {value.dtype}"

        assert (
            len(value.shape) == 2
        ), f"expected shape: (N,1), found {value.shape}"

        assert (
            value.shape[1] == 1
        ), f"expected one-dimensional values, found {value.shape}"

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def cdf(self, value):
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the p-quantile for the distribution.

        :param p: the parameter p of the quantile

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same

        """
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )

    @property
    def parameter(self) -> torch.Tensor:
        raise NotImplementedError(
            "You should subclass Distribution and " "implement this method."
        )


class Exponential(ContinuousDistribution):
    def __init__(self, rate: float, boundary: float = None):
        super().__init__()
        if boundary is not None:
            self.boundary = Parameter(
                torch.tensor([boundary]).to(torch.get_default_dtype()),
                requires_grad=False,
            )
        else:
            self.boundary = None

        self._rate = Parameter(
            inv_softplus(torch.tensor([rate]).to(torch.get_default_dtype())),
            requires_grad=True,
        )

    def _validate_args(self, value):
        super()._validate_args(value)

        assert torch.all(value >= 0), (
            f"Input values cannot be smaller"
            f" than 0. Rate is {self.rate} and values are {value}"
        )

    @property
    def rate(self) -> torch.Tensor:
        r = softplus(self._rate) + 1e-32

        if self.boundary > r:
            self._rate.data = inv_softplus(self.boundary.data)
            r = softplus(self._rate) + 1e-32
            return r

        return r

    @property
    def parameter(self) -> torch.Tensor:
        return self.rate

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.rate) - self.rate * value

    def cdf(self, value):
        #one = torch.tensor([1.0]).to(self.device)
        return self._one - torch.exp(-self.rate * value)

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        # t = time.time()

       # one = torch.tensor([1.0], device=self.device)#.to(self.device)
        q = -torch.log(self._one - p) / self.rate

        # q_time = time.time() - t
        # print(f'Compute quantile of original distribution took {q_time:.5f}')
        return q, q


class Uniform(ContinuousDistribution):
    def __init__(
        self,
        b: float,
    ):
        super().__init__()
        self.b = Parameter(
            torch.tensor([b]).to(torch.get_default_dtype()),
            requires_grad=False,  # DO NOT LEARN THIS
        )

    def _validate_args(self, value):
        super()._validate_args(value)

        assert torch.all(value >= 0), (
            f"Input values cannot be smaller" f" than 0."
        )

    @property
    def parameter(self) -> torch.Tensor:
        return self.b

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.b - 1.0)

    def cdf(self, value):
        #one = torch.tensor([1.0]).to(self.device)
        return (value - self._one) / (self.b - self._one)

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.b
        return b, b


class PowerLaw(ContinuousDistribution):
    def __init__(
        self, gamma: float, xmin: float = 1.0, low_deg_sat: float = 0.0,
            boundary: float = None
    ):
        super().__init__()
        assert xmin >= 0

        if boundary is not None:
            self.boundary = Parameter(
                torch.tensor([boundary]).to(torch.get_default_dtype()),
                requires_grad=False,
            )
        else:
            self.boundary = None

        self._xmin = Parameter(
            torch.tensor([xmin]).to(torch.get_default_dtype()),
            requires_grad=False,
        )
        self._gamma = Parameter(
            torch.tensor([gamma]).to(torch.get_default_dtype()),
            requires_grad=True,
        )
        self._low_deg_sat = Parameter(
            torch.tensor([low_deg_sat]).to(torch.get_default_dtype()),
            requires_grad=False,
        )
        

    def _validate_args(self, value):
        super()._validate_args(value)
        assert torch.all(value >= 0.0), (
            f"Input values cannot be smaller" f" than the scale {self.xm}."
        )

    @property
    def gamma(self) -> torch.Tensor:
        g = relu(self._gamma)
        if self.boundary is None:
            return g
        else:
            if self.boundary > g:
                self._gamma.data = self.boundary.data.clone()
                g = relu(self._gamma) + 1e-32
                return g
            else:
                return g


    @property
    def low_deg_sat(self) -> torch.Tensor:
        return self._low_deg_sat

    @property
    def parameter(self) -> torch.Tensor:
        return self.gamma

    @property
    def xmin(self) -> torch.Tensor:
        return self._xmin

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # p(k) = C k^{-gamma}, C = (gamma-1)k_min^{gamma-1}
        assert torch.all(value >= 0.0)
        #one = torch.tensor([1.0]).to(self.device)
        xmin = self._one * self.xmin
        # shift input of self.xmin so that passing value=0 corresponds to xmin
        value = value + xmin

        gamma = self.gamma
        gmo = self.gamma - self._one
        log_C = torch.log(gmo)

        log_p = (
            log_C
            + gmo * torch.log(self.xmin + self.low_deg_sat)
            - gamma * torch.log(value + self.low_deg_sat)
        )
        return log_p

    def cdf(self, value):
        assert torch.all(value >= 0.0)
        #one = torch.tensor([1.0]).to(self.device)
        xmin = self._one * self.xmin
        low_deg_sat = self.low_deg_sat
        # shift input of self.xmin so that passing value=0 corresponds to xmin
        value = value + xmin

        return self._one - torch.pow(
            (xmin + low_deg_sat) / (value + low_deg_sat), self.gamma - self._one
        )

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        #one = torch.tensor([1.0]).to(self.device)
        exp_term = -(self._one / (self.gamma - self._one))
        q = (self.xmin + self.low_deg_sat) * torch.pow(
            (self._one - p), exp_term
        ) - self.low_deg_sat
        return q, q


class DiscretizedDistribution(Module):
    def __init__(self, **kwargs):
        """
        Creates a discretized version of a continuous distribution such that

            p(x) = phi(x+1) - phi(x)

        where phi is the cdf of the original distribution.

        :param kwargs: a dictionary with a key 'base_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()
        base_d_cls, base_d_args = return_class_and_args(
            kwargs, "base_distribution"
        )
        self.base_distribution = base_d_cls(**base_d_args)
        self.device = "cpu"
        self.register_buffer('_one', torch.tensor(1.0, device=self.device))

    def to(self, device):
        super().to(device)
        self.device = device
        self._one.to(self.device)
        self.base_distribution.to(device)

    def get_q_named_parameters(self) -> dict:
        return self.base_distribution.get_q_named_parameters()

    def _validate_args(self, value):
        self.base_distribution._validate_args(value)

        # check values are integers
        assert torch.allclose(
            value, value.int().to(torch.get_default_dtype())
        ), f"expected float tensor with integer values, got {value}."

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the log pdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        # self._validate_args(value)

        #one = torch.ones(1).to(value.device)
        # avoids a degenerate case where the base distribution has the
        # same cdf for both value and value+1
        # which leads to nan. Also, a too small value can cause some
        # distributions to have prob 1 for a single neuron, and the model
        # gets trapped in there
        #tmp = torch.ones(1, device=value.device) * 1e-6
        return torch.log(
            self.base_distribution.cdf(value + self._one)
            - self.base_distribution.cdf(value)
            + 1e-6
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Computes the cdf of the distribution

        :param value: a tensor of shape Nx1, where N is the number of samples

        :return: a tensor of shape Nx1
        """
        # self._validate_args(value)
        #one = torch.ones(1, device=value.device)

        return self.base_distribution.cdf(value + self._one)

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated p-quantile for the discrete distribution.
        The lower and upper bounds returned by the method will coincide, since
        we provide the smallest integer x such that cdf(x) >= p

        :param p: the parameter p

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        lower_bound, upper_bound = self.base_distribution.quantile(p)

        # TODO TEST: see if we can avoid the call to CDF
        # corner case or case when quantile is known exactly
        if lower_bound == upper_bound:
            u = upper_bound.to(self.device)
            return u + 2, u + 2  # + 2 to avoid degenerate cases

        # Now perform binary search over the integers to find the smallest x
        # such that cdf(x) >= p. The boundaries of the search are given by the
        # bounds, and we use the fact that the cdf forms an ordered sequence
        l = torch.floor(lower_bound).to(self.device)
        u = torch.ceil(upper_bound).to(self.device)

        # ------------------------------------------------------------------ #
        # code that saves the day in case you implemented wrong bounds
        if self.cdf((u).unsqueeze(1)) < p:
            print(
                "WARNING: your upper bound on the quantile is "
                f"not working as expected:{(u, self.cdf((u).unsqueeze(1)))}"
            )
            ok = False
            while not ok:
                u += 1
                if self.cdf((u).unsqueeze(1)) >= p:
                    ok = True
            return u, u + 2  # + 2 to avoid degenerate cases
        # ------------------------------------------------------------------ #

        # if lower bound is already sufficient, stop, the normal and folded
        # normal curves are very similar at the desired quantile
        if self.cdf(l.unsqueeze(1)) >= p:
            # we could test l-1 (because of discretization), but in the end
            # it does not make a big difference
            assert self.cdf(l.unsqueeze(1)) >= p
            return l, l + 2  # + 2 to avoid degenerate cases

        # adapt the search: U will always have cdf(U) >= p, so we need to
        # check when we move from cdf(U) > p to cdf(U-1) <= p
        while l < u:
            if l == (u - 1.0):
                # assert self.cdf((u + 1).unsqueeze(1)) >= p
                # return u + 1, u + 1
                assert self.cdf(u.unsqueeze(1)) >= p
                return u, u + 2  # + 2 to avoid degenerate cases

            m = torch.floor((l + u) / 2.0)
            cdf_m = self.cdf(m.unsqueeze(1))

            if cdf_m < p:
                # move L to the right, closing the gap
                l = m + 1.0
            elif cdf_m > p:
                # move U to the left, closing the gap
                u = m - 1.0

    def compute_probability_vector(self, x) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        log_probs = self.log_prob(x).squeeze(1)
        probs = log_probs.exp()
        probs = probs / probs.sum()
        return probs

    @property
    def mean(self) -> torch.Tensor:
        return self.base_distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        return self.base_distribution.variance


class TruncatedDistribution(Module):
    def __init__(self, truncation_quantile: float, **kwargs):
        """
        Truncates a discretized distribution to a given quantile and
        renormalizes its probability.

        :param truncation_quantile: the quantile in [0,1] at which we want
            to truncate the discrete distribution.
        :param kwargs: a dictionary with a key 'discretized_distribution' that
            allows us to instantiate a discretized distribution
        """
        super().__init__()

        dist_d_cls, dist_d_args = return_class_and_args(
            kwargs, "discretized_distribution"
        )
        self.discretized_distribution = dist_d_cls(**dist_d_args)
        self.truncation_quantile = truncation_quantile

        self.x = None        

    def to(self, device):
        super().to(device)
        self.device = device
        self.discretized_distribution.to(device)

    def get_q_named_parameters(self) -> dict:
        return self.discretized_distribution.get_q_named_parameters()

    def compute_truncation_number(self) -> int:
        """
        Computes the truncation number at the specified quantile.

        :return: a positive integer holding the truncation number

        """

        # exploits the implementation of quantile() for the
        # DiscretizedDistribution, which returns

        _, truncation_number = self.discretized_distribution.quantile(
            p=self.truncation_quantile
        )

        # detach: this must not be part of the gradient computation in any way
        truncation_number = truncation_number.int().item()

        return truncation_number

    def compute_probability_vector(self) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        truncation_number = self.compute_truncation_number()

        if self.x is None or self.x.size(0) != truncation_number or self.device != self.x.device:
            # no gradient so far, we detach on purpose
            self.x = torch.arange(
                truncation_number,
                dtype=torch.get_default_dtype(),
                device=self.device,
            ).unsqueeze(1)


        probs = self.discretized_distribution.compute_probability_vector(self.x)

        #assert torch.allclose(probs.sum(), torch.ones(1, device=self.device)),(
        #    probs, self.discretized_distribution.base_distribution.k, self.discretized_distribution.base_distribution.b)
        return probs
    
    def compute_probability_vector_known_width(self, width) -> torch.Tensor:
        """
        Computes the **renormalized** vector of probabilities on the fly

        :return: a vector of arbitrary length with the probabilities
        """
        if self.x is None or self.x.size(0) != width or self.device != self.x.device:
            # no gradient so far, we detach on purpose
            self.x = torch.arange(
                width,
                dtype=torch.get_default_dtype(),
                device=self.device,
            ).unsqueeze(1)


        probs = self.discretized_distribution.compute_probability_vector(self.x)

        #assert torch.allclose(probs.sum(), torch.ones(1, device=self.device)),(
        #    probs, self.discretized_distribution.base_distribution.k, self.discretized_distribution.base_distribution.b)
        return probs

    @property
    def mean(self) -> torch.Tensor:
        proba = self.compute_probability_vector()
        return (proba * torch.arange(len(proba)).to(proba.device)).sum()

    def quantile(self, p: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated p-quantile for the discrete distribution.
        The lower and upper bounds returned by the method will coincide, since
        we provide the smallest integer x such that cdf(x) >= p

        :param p: the parameter p

        :return: lower and upper bounds for the p-quantile. If the p-quantile
            can be computed exactly then they are the same
        """
        return self.discretized_distribution.quantile(p)
