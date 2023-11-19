from absl import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver import ChromeOptions

from pathlib import Path
import time

from handle import play

URL_BASE = "https://ebird.org/explore"


def find_images(specie, path_dir: Path, wait_time=2, show_more=5, num_threads=5):
    t_s = time.time()
    options = ChromeOptions()
    # options.add_argument("--headless=new")
    with webdriver.Chrome(options=options) as driver:
        driver.get(URL_BASE)
        driver.maximize_window()

        wait = WebDriverWait(driver, 30)

        # 1
        driver.find_element(By.ID, "Suggest-0").click()

        # 2 | type | id=Suggest-0 | Blue Jay
        driver.find_element(By.ID, "Suggest-0").send_keys(specie)

        # 3 | click | css=.Suggestion-text |
        wait.until(lambda d: d.find_element(By.CSS_SELECTOR, ".Suggestion-text"))
        driver.find_element(By.CSS_SELECTOR, ".Suggestion-text").click()

        # # 4 | runScript | window.scrollTo(0,400) |
        driver.execute_script("window.scrollTo(0,600)")

        # 5 | click | linkText=View all |
        driver.find_element(By.LINK_TEXT, "View all").click()

        # 6 | click | id=show_more |
        for i in range(0, show_more):
            time.sleep(wait_time)
            driver.find_element(By.CSS_SELECTOR, ".pagination > .Button").click()

        logging.info("Handing over the page source")
        with open(path_dir / 'page_src.log', 'w') as f:
            f.write(driver.page_source)
        play(driver.page_source, path_dir, num_threads=num_threads)

    logging.info(f"Execution time {time.time() - t_s}")
    logging.info(f"Scrapping done for {specie}")

    return 0
