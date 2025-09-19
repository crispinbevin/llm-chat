import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import re
from typing import List, Dict, Set
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from datetime import datetime

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100, delay: float = 1.0, use_selenium: bool = True):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict] = []
        self.use_selenium = use_selenium

        # Selenium setup
        if self.use_selenium:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")

        # FAISS setup
        self.dimension = 384
        self.index = None

    def fetch_html(self, url: str) -> str:
        """Fetch page HTML with Selenium (or fallback requests)"""
        try:
            if self.use_selenium:
                self.driver.get(url)
                time.sleep(2)  # let JS render
                return self.driver.page_source
            else:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        parsed = urlparse(url)
        if parsed.netloc != self.domain:
            return False
        skip_patterns = [
            '/wp-admin/', '/admin/', '/login/', '/register/',
            '.pdf', '.jpg', '.png', '.gif', '.css', '.js',
            '/api/', '/rss/', '/feed/', '#'
        ]
        return not any(pattern in url.lower() for pattern in skip_patterns)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]"\'/@#$%&*+=]', '', text)
        return text.strip()

    def extract_content(self, soup: BeautifulSoup) -> Dict:
        """Extract meaningful content from HTML"""
        for element in soup(['script', 'style', 'nav', 'header', 'footer',
                             'aside', 'noscript', 'iframe', 'form']):
            element.decompose()

        ad_classes = ['ad', 'advertisement', 'banner', 'popup', 'modal', 'cookie', 'social']
        for class_name in ad_classes:
            for element in soup.find_all(attrs={'class': re.compile(class_name, re.I)}):
                element.decompose()

        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.h1:
            title = soup.h1.get_text()

        main_content = ""
        content_selectors = ['main', 'article', '.content', '.post', '.entry']
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text()
                break
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()

        return {
            'title': self.clean_text(title),
            'content': self.clean_text(main_content)
        }

    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def crawl_page(self, url: str) -> List[Dict]:
        try:
            print(f"Crawling: {url}")
            html = self.fetch_html(url)
            if not html:
                return []
            soup = BeautifulSoup(html, 'html.parser')
            content_data = self.extract_content(soup)
            if not content_data['content']:
                print(f"No content found for {url}")
                return []
            chunks = self.chunk_text(content_data['content'])
            page_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'id': hashlib.md5(f"{url}_{i}".encode()).hexdigest(),
                    'url': url,
                    'title': content_data['title'],
                    'content': chunk,
                    'chunk_index': i,
                    'crawled_at': datetime.now().isoformat()
                }
                page_chunks.append(chunk_data)
            return page_chunks
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return []

    def find_links(self, url: str, soup: BeautifulSoup) -> Set[str]:
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.add(full_url)
        return links

    def crawl_website(self) -> List[Dict]:
        print(f"Starting crawl of {self.base_url}")
        urls_to_visit = [self.base_url]
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            self.visited_urls.add(current_url)
            page_chunks = self.crawl_page(current_url)
            self.crawled_data.extend(page_chunks)
            if len(self.visited_urls) < min(20, self.max_pages // 2):
                try:
                    html = self.fetch_html(current_url)
                    soup = BeautifulSoup(html, 'html.parser')
                    new_links = self.find_links(current_url, soup)
                    urls_to_visit.extend(list(new_links)[:5])
                except:
                    pass
            time.sleep(self.delay)
        print(f"Crawling completed. Found {len(self.crawled_data)} chunks from {len(self.visited_urls)} pages")
        return self.crawled_data

    def create_embeddings(self) -> np.ndarray:
        if not self.crawled_data:
            return np.array([])
        print("Creating embeddings...")
        texts = [chunk['content'] for chunk in self.crawled_data]
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
        return np.array(embeddings).astype('float32')

    def build_faiss_index(self, embeddings: np.ndarray):
        print("Building FAISS index...")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def save_data(self, output_dir: str = "crawl_data"):
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=2)
        if self.index:
            index_path = os.path.join(output_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
        info = {
            'base_url': self.base_url,
            'total_chunks': len(self.crawled_data),
            'total_pages': len(self.visited_urls),
            'crawled_at': datetime.now().isoformat(),
            'embedding_model': 'all-MiniLM-L6-v2'
        }
        info_path = os.path.join(output_dir, "crawl_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Data saved to {output_dir}/")

    def run_full_crawl(self, output_dir: str = "crawl_data"):
        self.crawl_website()
        if not self.crawled_data:
            print("No data crawled!")
            return
        embeddings = self.create_embeddings()
        self.build_faiss_index(embeddings)
        self.save_data(output_dir)
        if self.use_selenium:
            self.driver.quit()
        print("Crawling pipeline completed!")


class RAGQueryEngine:
    def __init__(self, data_dir: str = "crawl_data"):
        self.data_dir = data_dir
        self.metadata = []
        self.index = None
        self.embedding_model = None
        self.load_data()

    def load_data(self):
        try:
            metadata_path = os.path.join(self.data_dir, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            index_path = os.path.join(self.data_dir, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"Loaded {len(self.metadata)} chunks for querying")
        except Exception as e:
            print(f"Error loading data: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.index or not self.embedding_model:
            return []
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        top_k = min(top_k, self.index.ntotal)  # cap results to available vectors
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            result = self.metadata[idx].copy()
            result['similarity_score'] = float(score)
            results.append(result)
        return results


if __name__ == "__main__":
    crawler = WebCrawler("https://hyperblox.io/", max_pages=20, delay=2, use_selenium=True)
    crawler.run_full_crawl()
    query_engine = RAGQueryEngine()
    results = query_engine.search("What is hyperblox about?", top_k=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['similarity_score']:.3f})")
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content'][:200]}...")
