# Importing necessary libraries
from math import floor
from pathlib import Path
from random import seed, shuffle
from shutil import move

from PIL import Image, ImageOps
from absl import app
from absl import flags

# Defining command-line flags
FLAGS = flags.FLAGS
flags.DEFINE_string("src_dir", None, "Source directory where images are stored.")
flags.DEFINE_string("dest_dir", None, "Destination directory to store processed images.")
flags.DEFINE_string("train_split", None, "Train split. [0-1]")

# Marking required flags
flags.mark_flags_as_required(["src_dir", "dest_dir", "train_split"])


# Function to resize and pad an image
def process(img: Image, size=(512, 512)) -> Image:
    # Convert image to RGB if it's in RGBA or P mode
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(size)
    delta_width = size[0] - img.size[0]
    delta_height = size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


# Function to split images into train and test sets
def train_test_split(dir_src, split):
    def _move(f, dest):
        move(f, dest / f.parts[-1])

    for dir_specie in (dir_src / 'tmp').glob('*'):
        if not dir_specie.is_dir():
            continue
        dir_train, dir_test = dir_src / 'train' / dir_specie.parts[-1], dir_src / 'test' / dir_specie.parts[-1]
        dir_train.mkdir(parents=True)
        dir_test.mkdir(parents=True)
        images = list(dir_specie.glob('*.jpg'))
        seed(44)
        shuffle(images)
        split_idx = floor(len(images) * split)
        train, test = images[:split_idx], images[split_idx:]
        [_move(f, dir_train) for f in train]
        [_move(f, dir_test) for f in test]
        dir_specie.rmdir()


# Main function for processing images
def _main(dir_src, dir_dest, split):
    if not dir_dest.exists():
        dir_dest.mkdir(parents=True)

    # Loop through source directory, process and move images
    for dir_specie in dir_src.glob('*'):
        if not dir_specie.is_dir():
            continue
        dir_specie_dest = dir_dest / 'tmp' / dir_specie.parts[-1]
        if not dir_specie_dest.exists():
            dir_specie_dest.mkdir(parents=True)
        for p_img in dir_specie.glob('*.jpg'):
            img = Image.open(p_img)
            file_name = p_img.parts[-1]
            img = process(img)
            img.verify()
            img.save(dir_specie_dest / file_name)

    # Split processed images into train and test sets
    train_test_split(dir_dest, split)
    (dir_dest / 'tmp').rmdir()


# Main function for command-line execution
def main(argv):
    del argv
    _main(Path(FLAGS.src_dir), Path(FLAGS.dest_dir), float(FLAGS.train_split))


# Execute main function
if __name__ == "__main__":
    app.run(main)

