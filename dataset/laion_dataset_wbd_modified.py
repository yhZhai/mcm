import base64
import binascii
import io
import itertools
import json
import math
import os
import pickle
from typing import List, Union

import albumentations as A
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import webdataset as wds
from braceexpand import braceexpand
from PIL import Image
from torch.utils.data import default_collate
from torchvision import transforms
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


def pilimg_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring, validate=True)
    except binascii.Error:
        jpgbytestring = imagestring
    # jpgbytestring = base64.b64decode(imagestring)
    image = Image.open(io.BytesIO(jpgbytestring))
    image = image.convert("RGB")
    return image


class WebdatasetFilter:
    def __init__(self, min_size=1024, max_pwatermark=0.5):
        self.min_size = min_size
        self.max_pwatermark = max_pwatermark

    def get_value(self, x, name, is_list=False):
        if is_list:
            return x.get(name)[0]
        else:
            return x.get(name)

    def __call__(self, x):
        try:
            if "json" in x:
                x_json = json.loads(x["json"])
                height = x_json.get("WIDTH", 0.0)
                is_list = True if isinstance(height, list) else False

                filter_size = (
                    self.get_value(x_json, "WIDTH", is_list) >= self.min_size
                    and self.get_value(x_json, "HEIGHT", is_list) >= self.min_size
                )

                filter_watermark = (
                    self.get_value(x_json, "pwatermark", is_list) <= self.max_pwatermark
                )

                return filter_size and filter_watermark
            else:
                return False
        except Exception:
            return False


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def decode_pkl():
    def _f(dictionary):
        new = {}
        for k, v in dictionary.items():
            if k.endswith(".pyd"):
                v = pickle.loads(v)
            new[k] = v
        return new

    return _f


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 512,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        min_size=0,
        pixel_mean: List = [0.5, 0.5, 0.5],
        pixel_std: List = [0.5, 0.5, 0.5],
    ):
        self.resolution = resolution
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [
                list(braceexpand(urls)) for urls in train_shards_path_or_url
            ]
            # flatten list using itertools
            train_shards_path_or_url = list(
                itertools.chain.from_iterable(train_shards_path_or_url)
            )
        if not train_shards_path_or_url.endswith(".tar"):
            files = os.listdir(train_shards_path_or_url)
            print(f"Extracting TAR files from {train_shards_path_or_url}")
            train_shards_path_or_url = [
                os.path.join(train_shards_path_or_url, f)
                for f in files
                if f.endswith(".tar")
            ]
        print(f"Using {len(train_shards_path_or_url)} TAR files")

        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=resolution, width=resolution, scale=(0.5, 1.0), p=1
                ),
                # A.Normalize(mean=self.pixel_mean, std=self.pixel_std, p=1, max_pixel_value=255),
                # ToTensorV2(p=1),
            ],
        )
        self.normalization = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.pixel_mean, std=self.pixel_std),
            ]
        )

        def transform(example):
            image = example["image"]
            image = pilimg_from_base64(image)
            image = self.transform(image=np.array(image))["image"]
            image = self.normalization(image)

            example["image"] = image
            example["text"] = (
                example["text"]
                .decode("utf-8")
                .split("<end_of_text>")[0]
                .replace("<start_of_text>", "")
            )
            return example

        processing_pipeline = [
            # wds.decode("pil", handler=wds.warn_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                text="text;txt;caption",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys({"image", "text"})),
            wds.map(transform),
            wds.to_tuple("image", "text"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * num_workers)
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == "__main__":
    pass
