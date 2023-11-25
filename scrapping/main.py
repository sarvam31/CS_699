import time
from pathlib import Path  # Importing the Path class from the pathlib module

from absl import app
from absl import flags
from absl import logging

from scrap_ebird import find_images

FLAGS = flags.FLAGS
flags.DEFINE_list("species", None, "Bird species to scrap.")  # Defining a command-line flag for bird species
flags.DEFINE_string("dest_dir", None, "Download directory.")  # Defining a command-line flag for download directory
flags.DEFINE_integer("wait_time", 2,
                     "Number of seconds to wait before clicking show more.")  # Defining a wait time flag
flags.DEFINE_integer("scrap_factor", 2, "Number of times to click show more.")  # Defining a scrap factor flag
flags.DEFINE_integer("num_threads", 5,
                     "Number of threads to use to retrieve download links.")  # Defining a threads flag

# Required flags
flags.mark_flags_as_required(["species", "dest_dir"])  # Marking certain flags as required


def _main(species, base_path: Path, wait_time=2, scrap_factor=2, num_threads=5):
    """
    Main function to scrap images for specified bird species.

    Args:
    - species (list): List of bird species to scrap.
    - base_path (Path): Path to the base directory.
    - wait_time (int): Number of seconds to wait before clicking 'show more'.
    - scrap_factor (int): Number of times to click 'show more'.
    - num_threads (int): Number of threads to use for downloading.

    Returns:
    - None
    """
    t_s = time.time()  # Record start time
    for i, specie in enumerate(species):
        specie = specie.strip()  # Remove leading/trailing spaces
        logging.info(f"{i} {specie}")  # Log the current species being processed
        path_specie = base_path / specie  # Generate path for the current species
        logging.info(f"Is directory {str(path_specie)} exists: {path_specie.exists()}")  # Log directory existence
        if not path_specie.exists():
            logging.info(f"Creating directory {path_specie}")  # Log directory creation
            path_specie.mkdir(parents=True)  # Create the directory
        logging.info(f"Started scrapping of {specie} images")  # Log the start of scraping images
        find_images(specie, path_specie, wait_time=wait_time, show_more=scrap_factor, num_threads=num_threads)
        # Call the find_images function to scrap images for the current species
    logging.info(f"Overall execution time {time.time() - t_s}")  # Log the overall execution time


def main(argv):
    """
    Main function to execute the image scraping.

    Args:
    - argv: Command line arguments

    Returns:
    - None
    """
    del argv  # Delete the argument vector
    _main(FLAGS.species, Path(FLAGS.dest_dir), wait_time=FLAGS.wait_time, scrap_factor=FLAGS.scrap_factor,
          num_threads=FLAGS.num_threads)  # Call the main function with flag values


if __name__ == "__main__":
    app.run(main)  # Run the main function when the script is executed directly
