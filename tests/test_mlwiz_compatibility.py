"""Compatibility checks against the installed MLWiz and its real CLI entry points."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Batch, Data

import yaml
from mlwiz.config_loader import load_config, load_experiment_config
from mlwiz.data.splitter import Splitter
from mlwiz.evaluation.grid import Grid
from mlwiz.evaluation.random_search import RandomSearch
from mlwiz.log.logger import Logger
from mlwiz.util import s2c

ROOT = Path(__file__).resolve().parents[1]


def search_config(path):
    config = load_experiment_config(path)
    return RandomSearch(config) if "random" in config else Grid(config)


class TestRepositoryConfigs(unittest.TestCase):
    def test_all_experiment_configs_expand_and_resolve_components(self):
        component_keys = {
            "class_name", "model", "dynamic_architecture", "loss", "scorer",
            "optimizer", "scheduler", "engine", "early_stopper", "plotter",
            "gradient_clipper", "main_scorer", "sample_method", "activation",
            "activation_outer", "readout", "experiment", "dataset_class",
            "dataset_getter", "data_loader", "optimizer_class_name",
            "scheduler_class_name",
        }

        def resolve(value, key=None):
            if isinstance(value, dict):
                for child_key, child in value.items():
                    resolve(child, child_key)
            elif isinstance(value, list):
                for child in value:
                    resolve(child, key)
            elif isinstance(value, str) and key in component_keys:
                self.assertTrue(callable(s2c(value)), value)

        for path in sorted((ROOT / "EXP_CONFIGS").rglob("*.yml")):
            with self.subTest(config=str(path.relative_to(ROOT))):
                search = search_config(path)
                self.assertGreater(len(search.hparams), 0)
                resolve(search.configs_dict)
                splits = ROOT / search.configs_dict["dataset"]["data_splits_file"]
                self.assertTrue(splits.is_file(), str(splits))
                self.assertTrue(splits.name.startswith(search.dataset_class.rsplit(".", 1)[-1] + "_"))
                self.assertGreater(Splitter.load(splits).n_outer_folds, 0)
                if "random" in search.configs_dict:
                    self.assertEqual(len(search.hparams), search.budget)

    def test_cached_scalar_targets_can_still_be_split(self):
        from dataset import DoubleMoon

        with tempfile.TemporaryDirectory(prefix="awn-labels-") as folder:
            dataset = DoubleMoon(storage_folder=folder, num_samples=2)
            old_samples = [(x, y.squeeze()) for x, y in dataset]
            dataset._save_dataset(old_samples, dataset.dataset_filepath)
            restored = DoubleMoon(storage_folder=folder, num_samples=2)
            splitter = Splitter(1, 1, 42, stratify=True, shuffle=True)
            has_targets, targets = splitter.get_targets(restored)
            self.assertTrue(has_targets)
            self.assertEqual(targets.tolist(), [0, 0, 1, 1])

    def test_all_data_configs_resolve_dataset_and_splitter(self):
        for path in sorted((ROOT / "DATA_CONFIGS").glob("*.yml")):
            with self.subTest(config=path.name):
                config = load_config(path)
                for key in ("dataset", "splitter"):
                    self.assertTrue(callable(s2c(config[key]["class_name"])))


class TestMLWizCLI(unittest.TestCase):
    def run_cli(self, command, config_file, debug=True):
        env = dict(
            os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
            RAY_TMPDIR=str(config_file.parent / "ray"),
            # These local workers share the synchronized environment and cwd.
            RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
        )
        result = subprocess.run(
            [str(Path(sys.executable).parent / command), "--config-file", str(config_file)]
            + (["--debug"] if command == "mlwiz-exp" and debug else []),
            cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("Traceback (most recent call last)", result.stdout)
        return result.stdout

    def test_data_and_short_experiments(self):
        # Temporary copies keep published splits, datasets and results untouched.
        with tempfile.TemporaryDirectory(prefix="awn-mlwiz-") as folder:
            tmp = Path(folder)
            data = load_config(ROOT / "DATA_CONFIGS/config_spiralHard.yml")
            data["dataset"]["args"]["storage_folder"] = str(tmp / "data")
            data["splitter"]["splits_folder"] = str(tmp / "splits")
            data_path = tmp / "data.yml"
            data_path.write_text(yaml.safe_dump(data))
            self.run_cli("mlwiz-data", data_path)
            # Check that cached data can be loaded as well as created.
            self.run_cli("mlwiz-data", data_path)

            for kind in ("fixed", "awn", "random", "local", "bo"):
                with self.subTest(experiment=kind):
                    source = "awn" if kind == "awn" else "fixed"
                    search = search_config(
                        ROOT / f"EXP_CONFIGS/GRID/MLP/SpiralHard/mlp_{source}.yml"
                    )
                    config = copy.deepcopy(search.configs_dict)
                    hparams = copy.deepcopy(search.hparams[0])
                    for key in list(hparams):
                        if key not in config["grid"]:
                            hparams.pop(key)
                    hparams.update(epochs=1, batch_size=512, checkpoint=False, plotter=None)
                    if kind in ("local", "bo"):
                        hparams.pop("num_hidden_neurons")
                        hparams.update(min_neurons=2, max_neurons=8)
                        hparams[f"{'local' if kind == 'local' else 'bo'}_search_budget"] = 1
                        config["experiment"]["experiment"] = (
                            "experiment.LocalSearchExperiment" if kind == "local"
                            else "experiment.BayesOptSearchExperiment"
                        )
                    config["grid"] = hparams
                    if kind == "random":
                        config["random"] = config.pop("grid")
                        config["random"]["budget"] = 1
                    config["dataset"]["storage_folder"] = str(tmp / "data")
                    config["dataset"]["data_splits_file"] = str(
                        tmp / "splits/SpiralHard/SpiralHard_outer1_inner1.splits"
                    )
                    config["resources"].update(device="cpu", max_cpus=1, max_gpus=0, gpus_per_task=0)
                    config["experiment"].update(
                        result_folder=str(tmp / "results"), exp_name=kind,
                        model_selection_training_runs=1, risk_assessment_training_runs=1,
                    )
                    config_path = tmp / f"{kind}.yml"
                    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                    output = self.run_cli("mlwiz-exp", config_path, debug=kind != "random")
                    assessment = tmp / f"results/{kind}_SpiralHard/MODEL_ASSESSMENT/assessment_results.json"
                    self.assertTrue(assessment.is_file(), output)
                    self.assertTrue(json.loads(assessment.read_text()), output)


class TestModelFamilies(unittest.TestCase):
    def test_one_epoch_on_synthetic_batches(self):
        cases = [
            ("CNN/CIFAR10", "resnet20", (32, 32, 3), 10),
            ("RNN", "rnn", (4, 4), 10),
            ("DGN/REDDIT_BINARY", "dgn", 1, 2),
            ("TRANSFORMER/Multi30K", "transformer", 4, 16),
        ]
        with tempfile.TemporaryDirectory(prefix="awn-models-") as folder:
            for family, stem, dim_input, dim_target in cases:
                for kind in ("fixed", "awn"):
                    with self.subTest(family=family, model=kind):
                        config = search_config(
                            ROOT / f"EXP_CONFIGS/GRID/{family}/{stem}_{kind}.yml"
                        ).hparams[0]
                        config.update(
                            device="cpu", checkpoint=False, plotter=None,
                            num_hidden_neurons=8,
                        )
                        if kind == "awn":
                            base = config["truncated_distribution"]["args"][
                                "discretized_distribution"
                            ]["args"]["base_distribution"]["args"]
                            base["rate"] = 0.5
                        if stem == "transformer":
                            config.update(
                                embedding_dim=8, num_attention_heads=2,
                                num_enc_dec_layers=1, num_hidden_layers=2,
                            )
                            inputs = torch.ones((2, 4, 2, 2), dtype=torch.long)
                            targets = torch.randint(0, dim_target, (2, 4))
                        elif stem == "dgn":
                            inputs = Batch.from_data_list([
                                Data(x=torch.ones(3, 1), edge_index=torch.tensor([
                                    [0, 1, 1, 2], [1, 0, 2, 1],
                                ])) for _ in range(2)
                            ])
                            targets = torch.tensor([0, 1])
                        else:
                            shape = (2, 4, 4) if stem == "rnn" else (2, 3, 32, 32)
                            inputs = torch.randn(shape)
                            targets = torch.tensor([0, 1])
                        path = Path(folder) / f"{stem}_{kind}"
                        experiment = s2c(config["experiment"])(config, str(path), 42)
                        logger = Logger(path / "experiment.log", "a", False)
                        model = experiment.create_model(dim_input, dim_target, config)
                        engine = experiment.create_engine(config, model)
                        batch = [(inputs, targets)]
                        results = engine.train(
                            batch, validation_loader=batch, test_loader=batch,
                            max_epochs=1, logger=logger,
                        )
                        self.assertEqual(len(results), 6)
                        for metrics in results:
                            for value in metrics.values():
                                self.assertTrue(torch.isfinite(torch.as_tensor(value)).all())


if __name__ == "__main__":
    unittest.main()
