import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from stem import Signal
from stem.control import Controller
import re
import time

class WebCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.session()
        self.proxies = {
            'http': 'socks5h://127.0.0.1:9050',
            'https': 'socks5h://127.0.0.1:9050'
        }
        self.session.proxies = self.proxies

    def renew_tor_ip(self):
        with Controller.from_port(port=9051) as controller:
            controller.authenticate(password='your_password')  # Set your Tor password here
            controller.signal(Signal.NEWNYM)
            time.sleep(10)  # Wait for the IP to change

    def fetch(self, url):
        headers = {'User-Agent': self.ua.random}
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    def extract_links(self, soup, base_url):
        links = set()
        for anchor in soup.find_all('a', href=True):
            link = anchor['href']
            if not re.match(r'http[s]?://', link):
                link = re.sub(r'^/', '', link)
                link = f'{base_url}/{link}'
            links.add(link)
        return links

    def crawl(self, start_url, depth=2):
        visited = set()
        to_visit = {start_url}

        for _ in range(depth):
            new_to_visit = set()
            for url in to_visit:
                if url in visited:
                    continue
                print(f'Crawling: {url}')
                html = self.fetch(url)
                if html:
                    soup = self.parse(html)
                    links = self.extract_links(soup, start_url)
                    new_to_visit.update(links)
                visited.add(url)
                self.renew_tor_ip()
            to_visit = new_to_visit - visited

        return visited

if __name__ == "__main__":
    start_url = "http://example.com"
    crawler = WebCrawler()
    visited_links = crawler.crawl(start_url)
    print(f"Visited {len(visited_links)} links.")

