# root/data_processing/internet_query.py
# Implements tools for querying and fetching data from the internet

import requests
from bs4 import BeautifulSoup
from googlesearch import search
import pandas as pd

class InternetQueryTool:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def search_google(self, query, num_results=10):
        return list(search(query, num_results=num_results))

    def fetch_webpage_content(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
        except requests.RequestException as e:
            print(f'Error fetching {url}: {e}')
            return None

    def search_and_summarize(self, query, num_results=5):
        results = self.search_google(query, num_results)
        summaries = []
        for url in results:
            content = self.fetch_webpage_content(url)
            if content:
                summary = content[:500] + '...' if len(content) > 500 else content
                summaries.append({
                    'url': url,
                    'summary': summary
                })
        return pd.DataFrame(summaries)

    def fetch_dataset(self, dataset_url):
        try:
            if dataset_url.endswith('.csv'):
                return pd.read_csv(dataset_url)
            elif dataset_url.endswith('.json'):
                return pd.read_json(dataset_url)
            elif dataset_url.endswith('.xlsx'):
                return pd.read_excel(dataset_url)
            else:
                print(f'Unsupported file format: {dataset_url}')
                return None
        except Exception as e:
            print(f'Error fetching dataset from {dataset_url}: {e}')
            return None

if __name__ == '__main__':
    query_tool = InternetQueryTool()
    results_df = query_tool.search_and_summarize('machine learning latest developments')
    print(results_df)
    
    dataset_url = 'https://example.com/dataset.csv'
    dataset = query_tool.fetch_dataset(dataset_url)
    if dataset is not None:
        print(dataset.head())
