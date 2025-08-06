from copy import deepcopy

import numpy as np
import torch
from mlwiz.evaluation.config import Config
from mlwiz.experiment import Experiment
from mlwiz.static import LOSS, MAIN_LOSS, SCORE, MAIN_SCORE
from mlwiz.training.engine import TrainingEngine
from skopt import gp_minimize
from skopt.space import Integer

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
from model import DynamicArchitecture


class WidthExperiment(Experiment):
    def create_engine(
        self,
        config: Config,
        model: DynamicArchitecture,
    ) -> TrainingEngine:
        """
        Sets the optimizer into the model to allow for dynamic optimization
        of newly inserted layers.
        """
        engine = super().create_engine(
            config,
            model,
        )
        model.set_optimizer(engine.optimizer)
        return engine


class LocalSearchExperiment(Experiment):
    """
    Performs a local search experiment as defined in https://arxiv.org/pdf/2005.02960
    Because local search has not been implemented in mlwiz, this experiment
    assumes that there is only one configuration specified in the configuration file,
    so that only run_test is called. Run_test performs a model selection using
    the value in the `min_neurons` and `max_neurons` keys before selecting the best
    results and running the final re-training.
    Note that this local search experiment focuses on the number of neurons only.

    IMPORTANT: Please look at results in the experiment.log file and not those returned by mlwiz
    """

    def run_valid(self, dataset_getter, logger):
        raise NotImplementedError("Please read the class documentation.")

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training, validation and test results
        for a `final run`.
        **Do not use the test to train the model
        nor for early stopping reasons!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training,validation,test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself with
            other fields to be used by the evaluator.
        """
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        dataset_getter.set_inner_k(0)
        dataset_getter.set_outer_k(0)

        config = dict(self.model_config)

        ### ------------------ LOCAL SEARCH PHASE ------------------------- ###
        higher_results_are_better = config["higher_results_are_better"]

        # Instantiate the Dataset
        train_loader = dataset_getter.get_inner_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_inner_val(
            batch_size=batch_size, shuffle=shuffle
        )

        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        min_neurons = config["min_neurons"]
        max_neurons = config["max_neurons"]

        assert (
            "num_hidden_neurons" not in config
        ), "You should not have num_hidden_neurons in your config for local search experiment!"
        local_search_budget = int(config["local_search_budget"])

        if higher_results_are_better:
            # we assume the score needs to be tracked
            best_score = float("-inf")
        else:
            # we assume the loss needs to be tracked
            best_score = float("inf")
        best_num_hidden_neurons = None

        logger.log(f"Starting local search experiment.")
        t = 0
        while t < local_search_budget:
            num_hidden_neurons = int(
                torch.randint(min_neurons, max_neurons, (1,))
            )
            config["num_hidden_neurons"] = num_hidden_neurons

            # Instantiate the Model
            model = self.create_model(dim_input_features, dim_target, config)

            # Instantiate the engine (it handles the training loop and the
            # inference phase by abstracting the specifics)
            training_engine = self.create_engine(config, model)

            (
                train_loss,
                train_score,
                _,  # check the ordering is correct
                val_loss,
                val_score,
                _,
                _,
                _,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=None,
                max_epochs=config["epochs"],
                logger=logger,
            )

            if higher_results_are_better:
                score = val_score["main_score"].item()
                if score > best_score:
                    best_score = score
                    best_num_hidden_neurons = num_hidden_neurons
                    logger.log(
                        f"Found new best score {score} at iteration {t}. "
                    )
                else:
                    logger.log(f"Reached local minimum, abort search. ")
                    break
            else:
                loss = val_loss["main_loss"].item()
                if loss < best_score:
                    best_score = loss
                    best_num_hidden_neurons = num_hidden_neurons
                    logger.log(
                        f"Found new best loss {loss} at iteration {t}. "
                    )
                else:
                    logger.log(f"Reached local minimum, abort search. ")
                    break

        num_hidden_neurons = best_num_hidden_neurons
        logger.log(f"Running final retrainings with {num_hidden_neurons}.")
        num_retrainings = 10  # HARDCODED

        config["num_hidden_neurons"] = num_hidden_neurons

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_outer_val(
            batch_size=batch_size, shuffle=shuffle
        )
        test_loader = dataset_getter.get_outer_test(
            batch_size=batch_size, shuffle=shuffle
        )

        # Call this after the loaders: the datasets may need to be instantiated
        # with additional parameters
        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        test_losses, test_scores = [], []
        for t in range(num_retrainings):
            # Instantiate the Model
            model = self.create_model(dim_input_features, dim_target, config)

            # Instantiate the engine (it handles the training loop and the
            # inference phase by abstracting the specifics)
            training_engine = self.create_engine(config, model)

            (
                train_loss,
                train_score,
                _,
                val_loss,
                val_score,
                _,
                test_loss,
                test_score,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=test_loader,
                max_epochs=config["epochs"],
                logger=logger,
            )

            test_losses.append(test_loss[MAIN_LOSS])
            test_scores.append(test_score[MAIN_SCORE])

            train_res = {LOSS: train_loss, SCORE: train_score}
            val_res = {LOSS: val_loss, SCORE: val_score}
            test_res = {LOSS: test_loss, SCORE: test_score}

        test_loss_mean, test_loss_std = np.mean(test_losses), np.std(
            test_losses
        )
        test_score_mean, test_score_std = np.mean(test_scores), np.std(
            test_scores
        )
        print(
            f"Exp ended with test loss {test_loss_mean} +- {test_loss_std} and test score {test_score_mean} +- {test_score_std}"
        )

        # THIS IS NOT RELEVANT FOR LOCAL SEARCH ANYMORE, WE TWEAKED MLWIZ
        # USE THE LOGGED STRING ABOVE
        return train_res, val_res, test_res


class BayesOptSearchExperiment(Experiment):
    """
    Performs a Bayesian Optimization search experiment using Expected Improvement
    as the acquisition function.
    Because BO search has not been implemented in mlwiz, this experiment
    assumes that there is only one configuration specified in the configuration file,
    so that only run_test is called. Run_test performs a model selection using
    the value in the `min_neurons` and `max_neurons` keys before selecting the best
    results and running the final re-training.
    Note that this local search experiment focuses on the number of neurons only.

    IMPORTANT: Please look at results in the experiment.log file and not those returned by mlwiz
    """

    def run_valid(self, dataset_getter, logger):
        raise NotImplementedError("Please read the class documentation.")

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training, validation and test results
        for a `final run`.
        **Do not use the test to train the model
        nor for early stopping reasons!**
        **If possible, rely on already available subclasses of this class**.

        It implements a simple training scheme.

        Args:
            dataset_getter (:class:`~mlwiz.data.provider.DataProvider`):
                a data provider
            logger (:class:`~mlwiz.log.logger.Logger`): the logger

        Returns:
            a tuple of training,validation,test dictionaries.
            Each dictionary has two keys:

            * ``LOSS`` (as defined in ``mlwiz.static``)
            * ``SCORE`` (as defined in ``mlwiz.static``)

            For instance, training_results[SCORE] is a dictionary itself with
            other fields to be used by the evaluator.
        """
        batch_size = self.model_config["batch_size"]
        shuffle = (
            self.model_config["shuffle"]
            if "shuffle" in self.model_config
            else True
        )

        dataset_getter.set_inner_k(0)
        dataset_getter.set_outer_k(0)

        config = dict(self.model_config)

        ### ------------------ LOCAL SEARCH PHASE ------------------------- ###
        higher_results_are_better = config["higher_results_are_better"]

        # Instantiate the Dataset
        train_loader = dataset_getter.get_inner_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_inner_val(
            batch_size=batch_size, shuffle=shuffle
        )

        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        min_neurons = config["min_neurons"]
        max_neurons = config["max_neurons"]

        assert (
            "num_hidden_neurons" not in config
        ), "You should not have num_hidden_neurons in your config for local search experiment!"
        bo_search_budget = int(config["bo_search_budget"])

        logger.log(f"Starting Bayesian Optimization experiment.")

        def objective(params):
            num_hidden_neurons = params[0]
            copy_config = deepcopy(config)
            copy_config["num_hidden_neurons"] = num_hidden_neurons

            # Instantiate the Model
            model = self.create_model(
                dim_input_features, dim_target, copy_config
            )

            # Instantiate the engine (it handles the training loop and the
            # inference phase by abstracting the specifics)
            training_engine = self.create_engine(copy_config, model)

            (
                train_loss,
                train_score,
                _,  # check the ordering is correct
                val_loss,
                val_score,
                _,
                _,
                _,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=None,
                max_epochs=copy_config["epochs"],
                logger=logger,
            )

            if higher_results_are_better:
                # assume accuracy
                return -float(val_score["main_score"])
            else:
                # assume loss
                return float(val_loss["main_loss"])

        # Perform Bayesian optimization
        result = gp_minimize(
            objective,  # Objective function
            [
                Integer(min_neurons, max_neurons, name="num_hidden_neurons")
            ],  # Hyperparameter space
            n_calls=bo_search_budget,  # Number of evaluations
            n_random_starts=1,
            random_state=int(torch.initial_seed()),  # Ensures reproducibility
        )

        num_hidden_neurons = result.x[0]
        logger.log(f"Best num_neurons found by BO are {num_hidden_neurons}.")
        if higher_results_are_better:
            logger.log(f"Best score found by BO are {-result.fun}.")
        else:
            logger.log(f"Best loss found by BO are {result.fun}.")

        logger.log(f"Running final retrainings with {num_hidden_neurons}.")
        num_retrainings = 10  # HARDCODED

        config["num_hidden_neurons"] = num_hidden_neurons

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(
            batch_size=batch_size, shuffle=shuffle
        )
        val_loader = dataset_getter.get_outer_val(
            batch_size=batch_size, shuffle=shuffle
        )
        test_loader = dataset_getter.get_outer_test(
            batch_size=batch_size, shuffle=shuffle
        )

        # Call this after the loaders: the datasets may need to be instantiated
        # with additional parameters
        dim_input_features = dataset_getter.get_dim_input_features()
        dim_target = dataset_getter.get_dim_target()

        test_losses, test_scores = [], []
        for t in range(num_retrainings):
            # Instantiate the Model
            model = self.create_model(dim_input_features, dim_target, config)

            # Instantiate the engine (it handles the training loop and the
            # inference phase by abstracting the specifics)
            training_engine = self.create_engine(config, model)

            (
                train_loss,
                train_score,
                _,
                val_loss,
                val_score,
                _,
                test_loss,
                test_score,
                _,
            ) = training_engine.train(
                train_loader=train_loader,
                validation_loader=val_loader,
                test_loader=test_loader,
                max_epochs=config["epochs"],
                logger=logger,
            )

            test_losses.append(test_loss[MAIN_LOSS])
            test_scores.append(test_score[MAIN_SCORE])

            train_res = {LOSS: train_loss, SCORE: train_score}
            val_res = {LOSS: val_loss, SCORE: val_score}
            test_res = {LOSS: test_loss, SCORE: test_score}

        test_loss_mean, test_loss_std = np.mean(test_losses), np.std(
            test_losses
        )
        test_score_mean, test_score_std = np.mean(test_scores), np.std(
            test_scores
        )
        print(
            f"Exp ended with test loss {test_loss_mean} +- {test_loss_std} and test score {test_score_mean} +- {test_score_std}"
        )

        # THIS IS NOT RELEVANT FOR BO SEARCH ANYMORE, WE TWEAKED MLWIZ
        # USE THE LOGGED STRING ABOVE
        return train_res, val_res, test_res
