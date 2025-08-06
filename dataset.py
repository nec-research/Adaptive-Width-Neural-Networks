import os
from pathlib import Path
from typing import Union, List, Tuple, Optional, Callable

import h5py
import numpy as np
import torch
import torchvision
from datasets import load_dataset
from mlwiz.data.dataset import DatasetInterface
from mlwiz.data.util import get_or_create_dir
from torch.nn.functional import pad
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from transformers import GPT2Tokenizer

import openml
from openml_pytorch import GenericDataset
from ucimlrepo import fetch_ucirepo 

class DoubleMoon(DatasetInterface):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.num_samples = kwargs.get("num_samples", 2500)  # for each class!
        self.radius = 10
        self.noise = 0.3
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 2

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        radius, n_samples, noise = self.radius, self.num_samples, self.noise
        np.random.seed(0)

        # Generate moon 1
        angle1 = np.random.rand(n_samples) * 180 * 2 * np.pi / 360
        x1 = (
            radius * np.cos(angle1)
            - 0.5 * radius
            + np.random.randn(n_samples) * noise
        )
        y1 = radius * np.sin(angle1) + np.random.randn(n_samples) * noise

        # Generate moon 2
        angle1 = np.random.rand(n_samples) * 180 * 2 * np.pi / 360
        x2 = (
            -(radius * np.cos(angle1) + np.random.randn(n_samples) * noise)
            + 0.5 * radius
        )
        y2 = (
            -(radius * np.sin(angle1) + np.random.randn(n_samples) * noise)
            + 0.5 * radius
        )

        # Combine moon 1 and moon 2 to form the dataset
        X = np.hstack([np.vstack([x1, y1]), np.vstack([x2, y2])])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

        print(y.shape)
        data_list = [
            (
                torch.tensor(X[:, i]).to(torch.get_default_dtype()),
                torch.tensor([y[i]]).long().long().squeeze(),
            )
            for i in range(2 * n_samples)
        ]

        return data_list


class Spiral(DatasetInterface):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.num_samples = kwargs.get("num_samples", 2500)  # for each class!
        self.repetition = kwargs.get("repetition", 1)
        self.radius = 10
        self.noise = 0.3
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 2

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        radius, n_samples, noise = self.radius, self.num_samples, self.noise
        np.random.seed(0)

        # generate first spiral
        n = (
            np.sqrt(np.random.rand(n_samples, 1))
            * self.repetition
            * 720
            * (2 * np.pi)
            / 360
        )
        d1x = -np.cos(n) * n + np.random.randn(n_samples, 1) * noise
        d1y = np.sin(n) * n + np.random.randn(n_samples, 1) * noise
        X1 = np.hstack((d1x, d1y))
        y1 = np.zeros((n_samples,))

        # generate second spiral
        n = (
            np.sqrt(np.random.rand(n_samples, 1))
            * self.repetition
            * 720
            * (2 * np.pi)
            / 360
        )
        d2x = -np.cos(n) * n + np.random.randn(n_samples, 1) * noise
        d2y = np.sin(n) * n + np.random.randn(n_samples, 1) * noise
        X2 = np.hstack((-d2x, -d2y))
        y2 = np.ones((n_samples,))

        X = np.vstack((X1, X2)) / self.repetition
        y = np.concatenate((y1, y2))

        data_list = [
            (
                torch.tensor(X[i, :]).to(torch.get_default_dtype()),
                torch.tensor([y[i]]).long().squeeze(),
            )
            for i in range(2 * n_samples)
        ]

        return data_list


class SpiralHard(Spiral):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
            num_samples=5000,
            repetition=2,
        )


class MNIST(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 28, 28, 1

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 10

    @property
    def _original_dataset_class(self) -> classmethod:
        return torchvision.datasets.MNIST

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        train = self._original_dataset_class(
            self.dataset_folder,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        test = self._original_dataset_class(
            self.dataset_folder,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        self.dataset = train + test
        return self.dataset


class CIFAR10(DatasetInterface):
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 32, 32, 3

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 10

    @property
    def _original_dataset_class(self) -> classmethod:
        return torchvision.datasets.CIFAR10

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        train = self._original_dataset_class(
            self.dataset_folder,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        test = self._original_dataset_class(
            self.dataset_folder,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        self.dataset = train + test
        return self.dataset


class CIFAR100(CIFAR10):

    @property
    def _original_dataset_class(self) -> classmethod:
        return torchvision.datasets.CIFAR100

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 100


class NCI1(DatasetInterface):

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        # on purpose, so we can fix the current issue with storing/saving with dill for Pytorch >2.4
        # super().__init__()
        self._name = self.__class__.__name__
        self._storage_folder = storage_folder
        self._raw_dataset_folder = raw_dataset_folder
        self.transform_train = transform_train
        self.transform_eval = transform_eval
        self.pre_transform = pre_transform
        self.dataset = None
        self._dataset_filename = f"{self.name}_processed_dataset.pt"

        # Create folders where to store processed dataset
        get_or_create_dir(self.dataset_folder)

        if self._raw_dataset_folder is not None and not os.path.exists(
            self.raw_dataset_folder
        ):
            raise FileNotFoundError(
                f"Folder {self._raw_dataset_folder} " f"not found"
            )

        # if any of the processed files is missing, process the dataset
        # and store the results in a file
        if not os.path.exists(Path(self.dataset_filepath)):
            print(
                f"File {self.dataset_filepath} from not found, "
                f"calling process_data()..."
            )
            dataset = self.process_dataset()

            # apply pre-processing if needed
            if self.pre_transform is not None:
                dataset = [self.pre_transform(d) for d in dataset]

            self.dataset = dataset

            # store dataset
            print(f"Storing into {self.dataset_filepath}...")
            torch.save(self.dataset, self.dataset_filepath)

        else:
            # Simply load the dataset in memory
            self.dataset = torch.load(self.dataset_filepath, weights_only=False)
    
    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 37

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        dataset = TUDataset(root=self.dataset_folder, name="NCI1")
        # casting class to int will allow PyG collater to create a tensor of
        # size (batch_size) instead of (batch_size, 1), making it consistent
        # with other non-graph datasets
        d = [(g, g.y.item()) for g in dataset]
        return d


class REDDIT_BINARY(NCI1):

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 1

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        dataset = TUDataset(root=self.dataset_folder, name="REDDIT-BINARY")
        # casting class to int will allow PyG collater to create a tensor of
        # size (batch_size) instead of (batch_size, 1), making it consistent
        # with other non-graph datasets
        d = []
        for g in dataset:
            g.x = torch.ones((g.num_nodes, 1))
            d.append((g, g.y.item()))
        return d


class Multi30K(DatasetInterface):

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
            **kwargs,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_len = kwargs["max_len"]

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return self.max_len

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return self.tokenizer.vocab_size

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """

        dataset = load_dataset("bentrevett/multi30k")

        print("Processing training set...")
        train = [(d["en"], d["de"]) for d in dataset["train"]]
        print("Processing validation set...")
        validation = [(d["en"], d["de"]) for d in dataset["validation"]]
        print("Processing test set...")
        test = [(d["en"], d["de"]) for d in dataset["test"]]

        self.dataset = train + validation + test
        return self.dataset

    def __getitem__(self, idx: int) -> object:
        r"""
        Returns sample ``idx`` of the dataset.

        Args:
            idx (int): the sample's index

        Returns: the i-th sample of the dataset

        """
        en, de = self.dataset[idx]
        en_tok = self.tokenizer(
            en, truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        de_tok = self.tokenizer(
            de, truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        en_input, en_att = en_tok["input_ids"].squeeze(0), en_tok[
            "attention_mask"
        ].squeeze(0)
        de_input, de_att = de_tok["input_ids"].squeeze(0), de_tok[
            "attention_mask"
        ].squeeze(0)

        # shift target to left (ignore first token)
        target = de_input[1:]

        # ignore last token so that dimensions match with target
        de_input, de_att = de_input[:-1], de_att[:-1]

        assert target.shape == de_input.shape and target.shape == de_att.shape

        # Pad sequences dynamically based on the max length in the batch
        en_tok_padded = pad(
            en_input,
            pad=(
                0,
                self.max_len - en_input.shape[0],
            ),
            mode="constant",
            value=0,
        )
        en_att_padded = pad(
            en_att,
            pad=(
                0,
                self.max_len - en_att.shape[0],
            ),
            mode="constant",
            value=0,
        )

        de_tok_padded = pad(
            de_input,
            pad=(
                0,
                self.max_len - de_input.shape[0],
            ),
            mode="constant",
            value=0,
        )
        de_att_padded = pad(
            de_att,
            pad=(
                0,
                self.max_len - de_att.shape[0],
            ),
            mode="constant",
            value=0,
        )

        tok_padded = torch.stack([en_tok_padded, de_tok_padded], dim=-1)
        att_padded = torch.stack([en_att_padded, de_att_padded], dim=-1)

        input_transformer = torch.stack([tok_padded, att_padded], dim=-1)

        target_padded = pad(
            target,
            pad=(
                0,
                self.max_len - target.shape[0],
            ),
            mode="constant",
            value=0,
        )

        # NOTE runtime preprocessing is handled by DataProvider
        return input_transformer, target_padded


class PermutedMNIST(DatasetInterface):
    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.permutation_seed = kwargs.get("permutation_seed", 42)
        np.random.seed(self.permutation_seed)
        self.permutation = np.random.permutation(28 * 28)

        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 28, 28

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 10

    @property
    def _original_dataset_class(self) -> classmethod:
        return torchvision.datasets.MNIST

    def process_dataset(self) -> List[object]:
        r"""
        Processes the dataset to the :obj:`self.dataset_folder` folder. It
        should generate files according to the obj:`self.dataset_file_names`
        list.
        """
        train = self._original_dataset_class(
            self.dataset_folder,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    self._apply_permutation,

                ]
            ),
        )
        test = self._original_dataset_class(
            self.dataset_folder,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    self._apply_permutation,

        ]
            ),
        )
        self.dataset = train + test
        return self.dataset

    def _apply_permutation(self, img: torch.Tensor) -> torch.Tensor:
        r"""
        Applies a fixed pixel permutation to the input image.

        Args:
            img (torch.Tensor): The input image as a tensor of shape (28, 28).

        Returns:
            torch.Tensor: The permuted image as a tensor of shape (28, 28).
        """
        img = img.view(-1)  # Flatten the image to 1D
        permuted_img = img[self.permutation]  # Apply the permutation
        return permuted_img.view(28, 28)  # Reshape back to 2D

    
class OpenMLPOL(DatasetInterface):
    dataset_id = 722  # POL dataset ID

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 48

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        # Get dataset by ID and split into train and test
        dataset = openml.datasets.get_dataset(self.dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X.to_numpy(dtype=np.float32)  
        y = y.cat.codes.to_numpy(dtype=np.int64) 

        data_list = [
            (
                torch.tensor(X[i, :]).to(torch.get_default_dtype()),
                torch.tensor([y[i]]).long().squeeze().unsqueeze(0),
            )
            for i in range(X.shape[0])
        ]

        return data_list
      

class OpenMLMiniBooNE(DatasetInterface):
    dataset_id = 41150  # jannis dataset ID   

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 50

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 4

    def process_dataset(self) -> List[object]:
        # Get dataset by ID and split into train and test
        dataset = openml.datasets.get_dataset(self.dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = torch.tensor(X.to_numpy(dtype=np.float32))
        y = y.to_numpy(dtype=np.int64) 

        # standardize all features
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-3)

        data_list = [
            (
                torch.tensor(X[i, :]).to(torch.get_default_dtype()),
                torch.tensor([y[i]]).long().squeeze().unsqueeze(0),
            )
            for i in range(X.shape[0])
        ]

        return data_list
    

class CreditCards(DatasetInterface):

    def __init__(
        self,
        storage_folder: str,
        raw_dataset_folder: Optional[str] = None,
        transform_train: Optional[Callable] = None,
        transform_eval: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            storage_folder,
            raw_dataset_folder,
            transform_train,
            transform_eval,
            pre_transform,
        )

    @staticmethod
    def _save_dataset(dataset, dataset_filepath):
        torch.save(dataset, dataset_filepath)

    @staticmethod
    def _load_dataset(dataset_filepath):
        return torch.load(dataset_filepath, weights_only=False)

    @property
    def dim_input_features(self) -> Union[int, Tuple[int]]:
        return 23

    @property
    def dim_target(self) -> Union[int, Tuple[int]]:
        return 2

    def process_dataset(self) -> List[object]:
        # fetch dataset 
        default_of_credit_card_clients = fetch_ucirepo(id=350) 
        
        # data (as pandas dataframes) 
        X = torch.tensor(default_of_credit_card_clients.data.features.to_numpy(dtype=np.float32))
        y = torch.tensor(default_of_credit_card_clients.data.targets.to_numpy(dtype=np.int64))


        # standardize all features
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-3)
        
        data_list = [
            (
                X[i],
                y[i].long(),
            )
            for i in range(X.shape[0])
        ]

        return data_list
