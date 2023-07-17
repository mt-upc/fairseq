#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Union
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
import soundfile as sf
from fairseq.data.audio.audio_utils import convert_waveform
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm


log = logging.getLogger(__name__)

TGT_SAMPLE_RATE = 16000
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)
        self.src_lang = source_language
        self.tgt_lang = target_language

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            download_url(covost_url, self.root.as_posix(), hash_value=None)
        extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        print("Checking audio files...")
        pool = Pool(processes=len(os.sched_getaffinity(0)))
        data = list(tqdm(pool.imap(self._process_element, data), total=len(data)))
        self.data = [e for e in data if e is not None]

    def _process_element(self, e):
        try:
            path = self.root / "clips" / e["path"]
            _ = torchaudio.info(path.as_posix())
            return e
        except RuntimeError:
            print(f"Skipping {e['path']}")
            return None

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def _convert_and_save(
    dataset: CoVoST, audio_root: Path, tgt_sample_rate: int, index: int
) -> None:
    waveform, sample_rate, _, _, _, utt_id = dataset[index]

    _wavform, _ = convert_waveform(
        waveform, sample_rate, to_mono=True,
        to_sample_rate=tgt_sample_rate
    )
    sf.write(
        (audio_root / f"{utt_id}.flac").as_posix(),
        _wavform.T.numpy(), tgt_sample_rate
    )
    
def _get_utt_manifest(
    dataset: CoVoST,
    audio_paths: dict[str, str],
    audio_lengths: dict[str, int],
    task: str,
    index: int
) -> dict[str, Union[str, int]]:
    _, _, src_utt, tgt_utt, speaker_id, utt_id = dataset[index]
    return {
        "id": utt_id,
        "audio": audio_paths[utt_id],
        "n_frames": audio_lengths[utt_id],
        "tgt_text": src_utt if task == "asr" else tgt_utt,
        "speaker": speaker_id,
        "tgt_lang": dataset.tgt_lang,
        "src_text": src_utt,
        "src_lang": dataset.src_lang,
    }


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    splits = ["train", "dev", "test"] if not args.no_train else ["dev", "test"]
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    # Extract features
    dir_name = "flac" if args.use_audio_input else "fbank80"
    dir_name += f"_{'-'.join(splits)}_{task}"
    audio_root = root / dir_name
    zip_path = root / f"{audio_root.name}.zip"
    if not zip_path.is_file():
        audio_root.mkdir(exist_ok=True)
        for split in splits:
            print(f"Fetching split {split}...")
            dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
            print("Convering and saving waveforms ...")
            if args.use_audio_input:
                _convert_and_save_ = partial(
                    _convert_and_save, dataset, audio_root, TGT_SAMPLE_RATE
                )
                num_cpus = len(os.sched_getaffinity(0))
                with Pool(num_cpus) as p:
                    _ = list(
                        tqdm(
                            p.imap(
                                _convert_and_save_,
                                range(len(dataset)),
                                chunksize=100
                            ),
                            total=len(dataset)
                        )
                    )
            else:
                print("Extracting filter bank features...")
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    extract_fbank_features(
                        waveform, sample_rate, audio_root / f"{utt_id}.npy"
                    )
        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(
        zip_path,
        is_audio=args.use_audio_input,
    )
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in splits:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        _get_utt_manifest_ = partial(
            _get_utt_manifest,
            dataset, audio_paths, audio_lengths, "st" if "st" in task else "asr"
        )
        num_processes = len(os.sched_getaffinity(0))
        with ThreadPool(num_processes) as p:
            manifest = list(
                tqdm(
                    p.imap(
                        _get_utt_manifest_,
                        list(range(len(dataset))),
                        chunksize=100
                    ),
                    total=len(dataset),
                )
            )
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(
            df,
            is_train_split=is_train_split,
            min_n_frames=(5 if not args.use_audio_input else 8000),
            max_n_frames=(3000 if not args.use_audio_input else 480_000),
        )
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")
    # Generate vocab
    if not args.no_train:
        vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size
            )
        # Generate config YAML
        gen_config_yaml(
            root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{task}.yaml",
            specaugment_policy="lb",
        )
    # Clean up
    shutil.rmtree(audio_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--use-audio-input", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
