from typing import Iterator, Tuple, Any

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class AssemblyRobotData(tfds.core.GeneratorBasedBuilder):
    """RLDS dataset builder for assembly_robot_data."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (features, etc.)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image0": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Front camera RGB observation (480x640x3).",
                                    ),
                                    "image1": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Side camera RGB observation (480x640x3).",
                                    ),
                                    "image_wrist": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist camera RGB observation (480x640x3).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state: 7-D vector (x, y, z, rx, ry, rz, gripper).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action: 7-D control vector (joint + gripper command).",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount; defaults to 1.0 if not provided.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward; often 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on first step of the episode.",
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode.",
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on terminal last step (True for demos).",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language instruction, repeated on every step.",
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="512-D language embedding (USE).",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the converted RLDS episode file.",
                            ),
                            "episode_id": tfds.features.Text(
                                doc='Unique episode identifier, e.g. "1118_155759".',
                            ),
                            "instruction": tfds.features.Text(
                                doc="Language instruction for this episode.",
                            ),
                            "num_steps": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Number of time steps in the episode.",
                            ),
                            "source_directory": tfds.features.Text(
                                doc="Original source directory of raw Gello pickles.",
                            ),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """Define train/val splits."""
        rlds_dir = "/scratch/pioneer/users/exr343/RLDS_2side_cleaned"

        episode_paths = sorted(glob.glob(os.path.join(rlds_dir, "*.pkl")))
        n = len(episode_paths)

        val_frac = 0.1  # 10% validation
        n_val = max(1, int(n * val_frac))

        train_paths = episode_paths[:-n_val]
        val_paths   = episode_paths[-n_val:]

        return {
            "train": self._generate_examples(train_paths),
            "val":   self._generate_examples(val_paths),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for a given split (train or val)."""

        def _parse_example(episode_path: str):
            with open(episode_path, "rb") as f:
                episode = pickle.load(f)

            meta = episode["episode_metadata"]
            data = episode["steps"]

            episode_steps = []
            n = len(data)
            for i, step in enumerate(data):
                obs = step["observation"]
                lang = step["language_instruction"]

                language_embedding = self._embed([lang])[0].numpy()

                episode_steps.append(
                    {
                        "observation": {
                            "image0": obs["image0"],              
                            "image1": obs["image1"],              
                            "image_wrist": obs["image_wrist"],    
                            "state": obs["state"],
                        },
                        "action": step["action"],
                        "discount": 1.0,
                        "reward": float(i == (n - 1)),
                        "is_first": (i == 0),
                        "is_last": (i == (n - 1)),
                        "is_terminal": (i == (n - 1)),
                        "language_instruction": lang,
                        "language_embedding":   language_embedding,
                    }
                )

            sample = {
                "steps": episode_steps,
                "episode_metadata": {
                    "file_path":        episode_path,
                    "episode_id":       meta["episode_id"],
                    "instruction":      meta["instruction"],
                    "num_steps":        meta["num_steps"],
                    "source_directory": meta["source_directory"],
                },
            }

            return episode_path, sample

        for episode_path in episode_paths:
            print("Processing", episode_path, flush=True)
            key, sample = _parse_example(episode_path)
            if sample is None:
                continue
            yield key, sample
