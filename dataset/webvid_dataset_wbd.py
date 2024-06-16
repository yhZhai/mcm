import io
import itertools
import json
import logging
import math
import os
import random
from typing import List, Union

import albumentations as A
import imageio.v3 as iio
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import default_collate
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

logger = logging.getLogger(__name__)


def unwrap_caption(sample):
    sample["text"] = sample["text"]["caption"]
    return sample


def recover_text_from_binary(sample):
    sample["text"] = sample["text"].decode("utf-8")
    sample["text"] = json.loads(sample["text"])
    return sample


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

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


class Text2VideoDataset:
    def __init__(
        self,
        train_shards_path_or_url: List[str],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        duration: int = 16,
        frame_interval: int = 8,
        frame_sel: str = "random",
        resolution: int = 512,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        pixel_mean: List = [0.5, 0.5, 0.5],
        pixel_std: List = [0.5, 0.5, 0.5],
    ):
        self.duration = duration
        self.frame_interval = frame_interval
        self.frame_sel = frame_sel
        self.resolution = resolution
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        whole_set = []
        for i, path in enumerate(train_shards_path_or_url):
            files = os.listdir(path)
            print(f"Extracting TAR files from {path}")
            subset = [os.path.join(path, f) for f in files if f.endswith(".tar")]
            whole_set.extend(subset)
        train_shards_path_or_url = whole_set
        print(f"Using {len(train_shards_path_or_url)} TAR files")

        additional_targets = {f"image{i}": "image" for i in range(1, self.duration)}
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=resolution, width=resolution, scale=(0.75, 1.0), p=1
                ),
                # A.Normalize(mean=self.pixel_mean, std=self.pixel_std, p=1, max_pixel_value=255),
                # ToTensorV2(p=1),
            ],
            additional_targets=additional_targets,
        )
        self.normalization = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.pixel_mean, std=self.pixel_std),
            ]
        )

        def sample_frames_from_binary(sample):
            reader = sample["video"][0].numpy()
            num_frames = reader.shape[0]

            start_frame_idx = 0
            if self.frame_sel == "random":
                max_start_frame = num_frames - self.duration * self.frame_interval
                start_frame_idx = random.randint(0, max_start_frame)

            frame_list = []
            for i in range(self.duration):
                index = start_frame_idx + i * self.frame_interval
                frame_list.append(reader[index])

            sample["video"] = frame_list
            return sample

        def transform_frame(sample):
            frame_list = sample["video"]
            transform_source = {"image": frame_list[0]}
            for i in range(1, self.duration):
                transform_source[f"image{i}"] = frame_list[i]
            transformed_frames = self.transform(**transform_source)
            frame_list = [self.normalization(transformed_frames["image"]).float()]
            frame_list.extend(
                [
                    self.normalization(transformed_frames[f"image{i}"]).float()
                    for i in range(1, self.duration)
                ]
            )
            frame_list = torch.stack(frame_list, dim=1)
            sample["video"] = frame_list
            return sample

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            # wds.SimpleShardList(train_shards_path_or_url),
            # wds.shuffle(100),
            # wds.split_by_node,
            # wds.split_by_worker,
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            wds.decode(wds.torch_video),
            wds.rename(
                video="mp4;avi;mov;mkv",
                text="json;txt;caption;text",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys({"video", "text"})),
            # wds.map(recover_text_from_binary),
            wds.map(sample_frames_from_binary),
            wds.map(transform_frame),
            wds.map(unwrap_caption),
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
        # self._train_dataset = wds.WebDataset(
        #     train_shards_path_or_url, nodesplitter=wds.split_by_node
        # )
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
