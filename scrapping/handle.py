from concurrent import futures
import multiprocessing as mp

from pathlib import Path

from bs4 import BeautifulSoup
import requests

from urllib3.util import parse_url

_img_count = 0


def get_links(soup):
    element = soup.find('img', class_="photo")
    return dict([l_s.strip().split(' ')[::-1] for l_s in element['srcset'].split(',')])


def retrieve_link(link):
    i_page = requests.get(link)
    soup_link = BeautifulSoup(i_page.content, features='html.parser')
    return get_links(soup_link).get("2400w")


def retrieve_links(links_images, links_download, num_threads=5):
    with futures.ThreadPoolExecutor(num_threads) as tex:
        tasks = [tex.submit(retrieve_link, link) for link in links_images]
        for f in futures.as_completed(tasks):
            links_download.put(f.result())


def download_images(urls, path_dir: Path):
    global _img_count
    while True:
        url = urls.get()
        _img_count += 1
        download_file(url, path_dir)
        print(f"No of images downloaded: {_img_count:04d}", end='\r', flush=True)
        urls.task_done()


def play(page_source, path_dir: Path, num_threads=5):
    global _img_count

    mp.freeze_support()
    downloads_queue = mp.JoinableQueue()
    downloads_proc = mp.Process(target=download_images, args=(downloads_queue, path_dir))
    downloads_proc.daemon = True
    downloads_proc.start()

    _img_count = 0

    soup = BeautifulSoup(page_source, features='html.parser')

    images_links = []
    image_rows = soup.find_all('div', class_='ResultsGallery-row')
    for i_r, e_r in enumerate(image_rows):
        for i, e in enumerate(e_r.children):
            images_links.append(e.attrs['href'])

    retrieve_links(images_links, downloads_queue, num_threads=num_threads)
    downloads_queue.join()
    print()


def download_file(url, path_dir: Path, file_name=None):
    response = requests.get(url, allow_redirects=True)
    p = Path(path_dir)
    if file_name:
        p = p / file_name
    else:
        p = p / f"{parse_url(url).path.split('/')[-2]}.jpg"
    with open(str(p), 'wb') as f:
        f.write(response.content)
