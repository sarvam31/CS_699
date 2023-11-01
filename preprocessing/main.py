from absl import app
from absl import flags

import os
from pathlib import Path
from shutil import move

from PIL import Image, ImageOps
from random import shuffle
from math import floor

FLAGS = flags.FLAGS
flags.DEFINE_string("src_dir", None, "Source directory where images are stored.")
flags.DEFINE_string("dest_dir", None, "Destination directory to store processed images.")
flags.DEFINE_string("train_split", None, "Train split. [0-1]")

# Required flags
flags.mark_flags_as_required(["src_dir", "dest_dir", "train_split"])


def process(img: Image, size=(512, 512)) -> Image:
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(size)
    delta_width = size[0] - img.size[0]
    delta_height = size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def train_test_split(dir_src, split):
    def _move(f, dest):
        move(f, dest / f.parts[-1])

    for dir_specie in dir_src.glob('*'):
        if not dir_specie.is_dir():
            continue
        dir_train, dir_test = dir_specie / 'train', dir_specie / 'test'
        os.mkdir(dir_train)
        os.mkdir(dir_test)
        images = list(dir_specie.glob('*.jpg'))
        shuffle(images)
        split_idx = floor(len(images) * split)
        train, test = images[:split_idx], images[split_idx:]
        [_move(f, dir_train) for f in train]
        [_move(f, dir_test) for f in test]


def _main(dir_src, dir_dest, split):
    if not dir_dest.exists():
        os.mkdir(dir_dest)

    for dir_specie in dir_src.glob('*'):
        if not dir_specie.is_dir():
            continue
        dir_specie_dest = dir_dest / dir_specie.parts[-1]
        if not dir_specie_dest.exists():
            os.mkdir(dir_specie_dest)
        for p_img in dir_specie.glob('*.jpg'):
            img = Image.open(p_img)
            file_name = p_img.parts[-1]
            img = process(img)
            img.save(dir_specie_dest / file_name)

    train_test_split(dir_dest, split)


def main(argv):
    del argv
    _main(Path(FLAGS.src_dir), Path(FLAGS.dest_dir), float(FLAGS.train_split))


if __name__ == "__main__":
    app.run(main)
