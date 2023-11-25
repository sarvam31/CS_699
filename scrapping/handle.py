from concurrent import futures
import multiprocessing as mp

from pathlib import Path

from bs4 import BeautifulSoup
import requests

from urllib3.util import parse_url  # Importing parse_url for URL parsing

_img_count = 0  # Global variable to track the number of downloaded images


def get_links(soup):
    # Function to extract image links from the HTML content
    element = soup.find('img', class_="photo")
    return dict([l_s.strip().split(' ')[::-1] for l_s in element['srcset'].split(',')])


def retrieve_link(link):
    # Function to retrieve specific image links based on provided URLs
    i_page = requests.get(link)
    soup_link = BeautifulSoup(i_page.content, features='html.parser')
    return get_links(soup_link).get("2400w")


def retrieve_links(links_images, links_download, num_threads=5):
    # Function to retrieve image links in parallel using ThreadPoolExecutor
    with futures.ThreadPoolExecutor(num_threads) as tex:
        tasks = [tex.submit(retrieve_link, link) for link in links_images]
        for f in futures.as_completed(tasks):
            links_download.put(f.result())


def download_images(urls, path_dir: Path):
    # Function to download images from provided URLs
    global _img_count
    while True:
        url = urls.get()
        _img_count += 1
        download_file(url, path_dir)
        print(f"No of images downloaded: {_img_count:04d}", end='\r', flush=True)
        urls.task_done()


def play(page_source, path_dir: Path, num_threads=5):
    # Main function to scrape image links and download images
    global _img_count

    mp.freeze_support()  # Ensuring Windows compatibility
    downloads_queue = mp.JoinableQueue()  # Creating a queue for image download
    downloads_proc = mp.Process(target=download_images, args=(downloads_queue, path_dir))
    downloads_proc.daemon = True  # Setting the process as a daemon
    downloads_proc.start()  # Starting the image download process

    _img_count = 0  # Resetting the image count

    soup = BeautifulSoup(page_source, features='html.parser')  # Creating a BeautifulSoup object from HTML content

    images_links = []
    image_rows = soup.find_all('div', class_='ResultsGallery-row')  # Finding image rows in the HTML
    for i_r, e_r in enumerate(image_rows):
        for i, e in enumerate(e_r.children):
            images_links.append(e.attrs['href'])  # Extracting image links from HTML elements

    retrieve_links(images_links, downloads_queue, num_threads=num_threads)  # Retrieving image links in parallel
    downloads_queue.join()  # Waiting for all downloads to complete
    print()


def download_file(url, path_dir: Path, file_name=None):
    # Function to download a file from a URL and save it to the specified directory
    response = requests.get(url, allow_redirects=True)  # Fetching the file content
    p = Path(path_dir)
    if file_name:
        p = p / file_name  # Saving with a specific file name if provided
    else:
        p = p / f"{parse_url(url).path.split('/')[-2]}.jpg"  # Extracting a filename from the URL
    with open(str(p), 'wb') as f:
        f.write(response.content)  # Writing the file content to the specified path
