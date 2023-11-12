from absl import app
from absl import flags
from absl import logging

import time
from pathlib import Path
import os

from scrap_ebird import find_images

FLAGS = flags.FLAGS
flags.DEFINE_list("species", None, "Bird species to scrap.")
flags.DEFINE_string("dest_dir", None, "Download directory.")
flags.DEFINE_integer("wait_time", 2, "Number of seconds to wait before clicking show more.")
flags.DEFINE_integer("scrap_factor", 2, "Number of times to click show more.")
flags.DEFINE_integer("num_threads", 5, "Number of threads to use to retrieve download links.")

# Required flags
flags.mark_flags_as_required(["species", "dest_dir"])


def _main(species, base_path: Path, wait_time=2, scrap_factor=2, num_threads=5):
    t_s = time.time()
    for i, specie in enumerate(species):
        specie = specie.strip()
        logging.info(f"{i} {specie}")
        path_specie = base_path / specie

        # TODO: create base path directory
        
        logging.info(f"Is directory {str(path_specie)} exists: {path_specie.exists()}")
        if not path_specie.exists():
            logging.info(f"Creating directory {path_specie}")
            os.mkdir(path_specie)
        logging.info(f"Started scrapping of {specie} images")
        find_images(specie, path_specie, wait_time=wait_time, show_more=scrap_factor, num_threads=num_threads)
    logging.info(f"Overall execution time {time.time() - t_s}")


def main(argv):
    del argv
    _main(FLAGS.species, Path(FLAGS.dest_dir), wait_time=FLAGS.wait_time, scrap_factor=FLAGS.scrap_factor,
          num_threads=FLAGS.num_threads)


if __name__ == "__main__":
    app.run(main)
