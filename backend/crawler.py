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

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100, delay: float = 1.0):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.crawled_data: List[Dict] = []
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
        # FAISS setup
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        parsed = urlparse(url)
        
        # Only crawl URLs from the same domain
        if parsed.netloc != self.domain:
            return False
            
        # Skip common non-content URLs
        skip_patterns = [
            '/wp-admin/', '/admin/', '/login/', '/register/',
            '.pdf', '.jpg', '.png', '.gif', '.css', '.js',
            '/api/', '/rss/', '/feed/', '#'
        ]
        
        return not any(pattern in url.lower() for pattern in skip_patterns)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]"\'/@#$%&*+=]', '', text)
        return text.strip()
    
    def extract_content(self, soup: BeautifulSoup) -> Dict:
        """Extract meaningful content from HTML"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'noscript', 'iframe', 'form']):
            element.decompose()
        
        # Remove elements with common ad/navigation classes
        ad_classes = ['nav', 'navigation', 'menu', 'sidebar', 'ad', 'advertisement', 
                     'banner', 'popup', 'modal', 'cookie', 'social']
        
        for class_name in ad_classes:
            for element in soup.find_all(attrs={'class': re.compile(class_name, re.I)}):
                element.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.h1:
            title = soup.h1.get_text()
        
        # Extract main content
        main_content = ""
        
        # Try to find main content areas
        content_selectors = ['main', 'article', '.content', '.post', '.entry']
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                main_content = content_elem.get_text()
                break
        
        # Fallback to body if no main content found
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()
        
        return {
            'title': self.clean_text(title),
            'content': self.clean_text(main_content)
        }
    
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks of roughly max_tokens"""
        # Simple word-based chunking (roughly 1 token = 1 word for estimation)
        words = text.split()
        chunks = []
        
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks
    
    def crawl_page(self, url: str) -> List[Dict]:
        """Crawl a single page and return chunks with metadata"""
        try:
            print(f"Crawling: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content_data = self.extract_content(soup)
            
            if not content_data['content']:
                print(f"No content found for {url}")
                return []
            
            # Create chunks
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
            print(f"Error crawling {url}: {str(e)}")
            return []
    
    def find_links(self, url: str, soup: BeautifulSoup) -> Set[str]:
        """Extract all valid links from a page"""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.add(full_url)
        
        return links
    
    def crawl_website(self) -> List[Dict]:
        """Main crawling function"""
        print(f"Starting crawl of {self.base_url}")
        
        urls_to_visit = [self.base_url]
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            # Crawl the page
            page_chunks = self.crawl_page(current_url)
            self.crawled_data.extend(page_chunks)
            
            # Find new links (only for the first few pages to avoid going too deep)
            if len(self.visited_urls) < min(20, self.max_pages // 2):
                try:
                    response = requests.get(current_url, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    new_links = self.find_links(current_url, soup)
                    urls_to_visit.extend(list(new_links)[:5])  # Limit new links per page
                except:
                    pass
            
            # Be polite
            time.sleep(self.delay)
        
        print(f"Crawling completed. Found {len(self.crawled_data)} chunks from {len(self.visited_urls)} pages")
        return self.crawled_data
    
    def create_embeddings(self) -> np.ndarray:
        """Create embeddings for all chunks"""
        if not self.crawled_data:
            return np.array([])
        
        print("Creating embeddings...")
        texts = [chunk['content'] for chunk in self.crawled_data]
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings")
        
        return np.array(embeddings).astype('float32')
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        print("Building FAISS index...")
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_data(self, output_dir: str = "crawl_data"):
        """Save crawled data and FAISS index"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=2)
        
        # Save FAISS index
        if self.index:
            index_path = os.path.join(output_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
        
        # Save crawl info
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
        """Run complete crawling pipeline"""
        # Crawl website
        self.crawl_website()
        
        if not self.crawled_data:
            print("No data crawled!")
            return
        
        # Create embeddings
        embeddings = self.create_embeddings()
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save everything
        self.save_data(output_dir)
        
        print("Crawling pipeline completed!")


# Utility class for querying the crawled data
class RAGQueryEngine:
    def __init__(self, data_dir: str = "crawl_data"):
        self.data_dir = data_dir
        self.metadata = []
        self.index = None
        self.embedding_model = None
        
        self.load_data()
    
    def load_data(self):
        """Load saved crawl data"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.data_dir, "metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load FAISS index
            index_path = os.path.join(self.data_dir, "faiss_index.bin")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print(f"Loaded {len(self.metadata)} chunks for querying")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index or not self.embedding_model:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    crawler = WebCrawler("https://hyperblox.io/", max_pages=50, delay=1.5)
    crawler.run_full_crawl()
    
    # Test querying
    query_engine = RAGQueryEngine()
    results = query_engine.search("What is hyperblox about?", top_k=3)
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (Score: {result['similarity_score']:.3f})")
        print(f"URL: {result['url']}")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content'][:200]}...")