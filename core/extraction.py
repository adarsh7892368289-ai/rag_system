"""
Universal Document Extractor - Fixed Production Version

Key fixes:
1. Proper document ID generation (no URL fragments)
2. Optimized metadata structure
3. Better unique ID generation with hashing
4. Merged chunk metadata for better searchability
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import scrapy
from scrapy.crawler import CrawlerProcess
import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation

from core.chunking import Chunker, Chunk
from config.settings import EXTRACTION, CHUNKING, ExtractionConfig
from utils.logger import get_logger, ProgressLogger, PerformanceLogger
from utils.validators import SourceValidator, FileValidator, URLValidator

logger = get_logger("extraction")

# Global storage for Scrapy results
SCRAPY_RESULTS = []


@dataclass
class Document:
    """Enhanced document structure with rich metadata"""
    content: str
    source: str
    doc_id: str
    source_type: str
    metadata: Dict
    chunk_id: Optional[int] = None
    chunk_metadata: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)
    
    def __repr__(self):
        return f"Document(source={self.source[:50]}, chunk_id={self.chunk_id}, words={len(self.content.split())})"


class TextSpider(scrapy.Spider):
    """Enhanced Scrapy spider with better error handling"""
    name = 'text_spider'
    
    def __init__(self, urls=None, follow_links=False, max_pages=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if isinstance(urls, list):
            self.start_urls = urls
        elif urls:
            self.start_urls = [urls]
        else:
            self.start_urls = []
        
        self.follow_links = follow_links
        self.max_pages = max_pages
        self.pages_scraped = 0
        self.visited_urls = set()
    
    def parse(self, response):
        """Parse webpage and extract text"""
        if self.pages_scraped >= self.max_pages:
            return
        
        if response.url in self.visited_urls:
            return
        
        self.visited_urls.add(response.url)
        self.pages_scraped += 1
        
        # Extract title
        title = response.css('title::text').get() or 'No title'
        
        # Extract text from multiple elements
        text_parts = []
        for selector in ['p::text', 'h1::text', 'h2::text', 'h3::text', 
                        'h4::text', 'li::text', 'td::text', 'span::text']:
            text_parts.extend(response.css(selector).getall())
        
        # Clean and join text
        text = ' '.join([t.strip() for t in text_parts if t.strip()])
        
        # Extract meta description
        meta_desc = response.css('meta[name="description"]::attr(content)').get() or ''
        
        SCRAPY_RESULTS.append({
            'content': text,
            'source': response.url,
            'source_type': 'web',
            'metadata': {
                'title': title.strip(),
                'meta_description': meta_desc.strip(),
                'url': response.url,
                'domain': urlparse(response.url).netloc,
                'extracted_at': datetime.now().isoformat(),
                'char_count': len(text),
                'word_count': len(text.split()),
                'status_code': response.status
            }
        })
        
        logger.info(f"   ✓ [{self.pages_scraped}/{self.max_pages}] "
                   f"{response.url[:60]}... ({len(text):,} chars)")
        
        # Follow links if enabled
        if self.follow_links and self.pages_scraped < self.max_pages:
            for link in response.css('a::attr(href)').getall():
                if link and not link.startswith(('#', 'javascript:', 'mailto:')):
                    # Stay within same domain
                    if urlparse(response.url).netloc in link or link.startswith('/'):
                        yield response.follow(link, callback=self.parse)


class DocumentExtractor:
    """
    Universal document extractor with smart chunking
    
    Features:
    - Multi-source extraction (web, PDF, Excel, Word, PPT, CSV, JSON, TXT)
    - Intelligent chunking with multiple strategies
    - Rich metadata extraction
    - Progress tracking and error handling
    - Batch processing support
    """
    
    def __init__(self,
                 chunker: Optional[Chunker] = None,
                 config: Optional[ExtractionConfig] = None):
        """
        Initialize extractor
        
        Args:
            chunker: Chunker instance (creates default if None)
            config: Extraction configuration (uses default if None)
        """
        self.chunker = chunker or Chunker()
        self.config = config or EXTRACTION
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Initialized DocumentExtractor")
        logger.info(f"  Output dir: {self.config.output_dir}")
        logger.info(f"  Chunking: {self.chunker.config.method}, "
                   f"{self.chunker.config.target_words}w")
    
    @staticmethod
    def generate_doc_id(source: str, chunk_index: Optional[int] = None) -> str:
        """
        Generate unique, clean document ID
        
        Uses hash-based ID to avoid special characters and ensure uniqueness
        
        Args:
            source: Source URL or file path
            chunk_index: Chunk index (None for un-chunked documents)
        
        Returns:
            Clean, unique document ID
        """
        # Create hash of source for uniqueness
        source_hash = hashlib.md5(source.encode()).hexdigest()[:12]
        
        # Extract readable identifier
        if URLValidator.is_url(source):
            parsed = urlparse(source)
            domain = parsed.netloc.replace('www.', '')
            # Get last meaningful path segment
            path_parts = [p for p in parsed.path.split('/') if p]
            identifier = path_parts[-1] if path_parts else domain
            identifier = re.sub(r'[^\w\-]', '_', identifier)[:30]
            base_id = f"web_{domain}_{identifier}_{source_hash}"
        else:
            filename = Path(source).stem
            # Handle Excel sheets
            if '::' in filename:
                filename = filename.replace('::', '_')
            identifier = re.sub(r'[^\w\-]', '_', filename)[:30]
            base_id = f"file_{identifier}_{source_hash}"
        
        # Add chunk index if provided
        if chunk_index is not None:
            return f"{base_id}_chunk{chunk_index}"
        else:
            return base_id
    
    def extract_multiple(self,
                        sources: List[str],
                        chunk: bool = True,
                        use_scrapy: bool = True,
                        follow_links: bool = False,
                        max_pages: int = 10,
                        parallel: bool = True) -> List[Document]:
        """
        Extract from multiple sources with parallel processing
        
        Args:
            sources: List of URLs or file paths
            chunk: Enable chunking
            use_scrapy: Use Scrapy for web scraping
            follow_links: Follow links when scraping
            max_pages: Max pages per seed URL
            parallel: Process files in parallel
        
        Returns:
            List of Document objects
        """
        # Validate sources
        validated_sources = SourceValidator.validate_sources(sources)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {len(sources)} source(s)")
        logger.info(f"{'='*70}")
        
        all_documents = []
        
        # Separate URLs and files
        urls = [s for s, t in validated_sources if t == 'url']
        files = [s for s, t in validated_sources if t == 'file']
        
        # Process URLs with Scrapy
        if urls:
            url_docs = self._process_urls(urls, chunk, use_scrapy, follow_links, max_pages)
            all_documents.extend(url_docs)
        
        # Process files
        if files:
            if parallel and len(files) > 1:
                file_docs = self._process_files_parallel(files, chunk)
            else:
                file_docs = self._process_files_sequential(files, chunk)
            all_documents.extend(file_docs)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Extraction complete: {len(all_documents)} document(s)")
        logger.info(f"{'='*70}\n")
        
        return all_documents
    
    def extract(self,
               source: str,
               chunk: bool = True,
               use_scrapy: bool = False,
               follow_links: bool = False,
               max_pages: int = 10) -> List[Document]:
        """
        Extract from a single source
        
        Args:
            source: URL or file path
            chunk: Enable chunking
            use_scrapy: Use Scrapy for web scraping
            follow_links: Follow links when scraping
            max_pages: Max pages per seed URL
        
        Returns:
            List of Document objects
        """
        source, source_type = SourceValidator.validate_source(source)
        
        logger.info(f"\nProcessing: {source}")
        
        with PerformanceLogger(f"Extraction from {Path(source).name if source_type == 'file' else source[:50]}"):
            if source_type == 'url':
                if use_scrapy:
                    raw_docs = self._extract_web_scrapy_batch([source], follow_links, max_pages)
                else:
                    raw_docs = self._extract_web_bs4(source)
            else:
                raw_docs = self._extract_file(source)
            
            documents = self._convert_to_documents(raw_docs, chunk)
        
        return documents
    
    def _process_urls(self, urls: List[str], chunk: bool, use_scrapy: bool,
                     follow_links: bool, max_pages: int) -> List[Document]:
        """Process URLs with Scrapy or BeautifulSoup"""
        mode = "crawling" if follow_links else "scraping"
        logger.info(f"\n🕷️  Scrapy {mode}: {len(urls)} seed URL(s)")
        if follow_links:
            logger.info(f"   Max pages per seed: {max_pages}")
        
        if use_scrapy:
            try:
                raw_docs = self._extract_web_scrapy_batch(urls, follow_links, max_pages)
                return self._convert_to_documents(raw_docs, chunk)
            except Exception as e:
                logger.error(f"❌ Scrapy failed: {e}")
                logger.info("   Falling back to BeautifulSoup...")
        
        # Fallback or direct BeautifulSoup
        all_docs = []
        for url in urls:
            try:
                raw_docs = self._extract_web_bs4(url)
                all_docs.extend(self._convert_to_documents(raw_docs, chunk))
            except Exception as e:
                logger.error(f"❌ {url}: {e}")
        
        return all_docs
    
    def _process_files_sequential(self, files: List[str], chunk: bool) -> List[Document]:
        """Process files sequentially"""
        logger.info(f"\n📁 Processing {len(files)} file(s) sequentially...")
        
        all_docs = []
        progress = ProgressLogger("File extraction", total=len(files))
        
        for idx, file_path in enumerate(files, 1):
            try:
                raw_docs = self._extract_file(file_path)
                docs = self._convert_to_documents(raw_docs, chunk)
                all_docs.extend(docs)
                progress.update(idx, f"{Path(file_path).name}: {len(docs)} docs")
            except Exception as e:
                logger.error(f"   ❌ {Path(file_path).name}: {e}")
                progress.update(idx, f"FAILED: {Path(file_path).name}")
        
        progress.complete(f"{len(all_docs)} documents extracted")
        return all_docs
    
    def _process_files_parallel(self, files: List[str], chunk: bool,
                               max_workers: int = 4) -> List[Document]:
        """Process files in parallel"""
        logger.info(f"\n📁 Processing {len(files)} file(s) in parallel (workers={max_workers})...")
        
        all_docs = []
        progress = ProgressLogger("File extraction", total=len(files))
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._extract_and_convert, file_path, chunk): file_path
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1
                
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    progress.update(completed, f"{Path(file_path).name}: {len(docs)} docs")
                except Exception as e:
                    logger.error(f"   ❌ {Path(file_path).name}: {e}")
                    progress.update(completed, f"FAILED: {Path(file_path).name}")
        
        progress.complete(f"{len(all_docs)} documents extracted")
        return all_docs
    
    def _extract_and_convert(self, file_path: str, chunk: bool) -> List[Document]:
        """Helper for parallel processing"""
        raw_docs = self._extract_file(file_path)
        return self._convert_to_documents(raw_docs, chunk)
    
    def _extract_file(self, file_path: str) -> List[Dict]:
        """Extract from file based on extension"""
        path, ext = FileValidator.validate_file_extension(file_path)
        
        extractors = {
            '.pdf': self._extract_pdf,
            '.xlsx': self._extract_excel,
            '.xls': self._extract_excel,
            '.csv': self._extract_csv,
            '.docx': self._extract_word,
            '.pptx': self._extract_pptx,
            '.txt': self._extract_txt,
            '.json': self._extract_json,
        }
        
        return extractors[ext](str(path))
    
    def _convert_to_documents(self, raw_docs: List[Dict], do_chunk: bool) -> List[Document]:
        """
        Convert raw extracted data to Document objects with optional chunking
        
        FIXED: Proper doc_id generation and merged metadata
        """
        documents = []
        
        for raw_doc in raw_docs:
            source = raw_doc['source']
            base_metadata = raw_doc['metadata'].copy()
            
            if do_chunk:
                # Chunk the content
                chunks = self.chunker.chunk(raw_doc['content'])
                
                # Add chunk summary to base metadata
                base_metadata['chunking'] = {
                    'method': self.chunker.config.method,
                    'target_words': self.chunker.config.target_words,
                    'overlap_words': self.chunker.config.overlap_words,
                    'total_chunks': len(chunks)
                }
                
                for chunk in chunks:
                    # Merge all metadata into one clean structure
                    chunk_metadata = {
                        **base_metadata,
                        'chunk_index': chunk.index,
                        'chunk_char_count': chunk.char_count,
                        'chunk_word_count': chunk.word_count,
                        'chunk_sentence_count': chunk.sentence_count,
                    }
                    
                    # Add chunk-specific metadata if available
                    if chunk.metadata:
                        chunk_metadata['chunk_keywords'] = chunk.metadata.get('keywords', [])
                        chunk_metadata['avg_sentence_length'] = chunk.metadata.get('avg_sentence_length', 0)
                    
                    # Generate clean doc_id
                    doc_id = self.generate_doc_id(source, chunk.index)
                    
                    doc = Document(
                        content=chunk.text,
                        source=source,
                        doc_id=doc_id,
                        source_type=raw_doc['source_type'],
                        metadata=chunk_metadata,
                        chunk_id=chunk.index,
                        chunk_metadata=None  # Merged into main metadata
                    )
                    documents.append(doc)
            else:
                # No chunking - single document
                doc_id = self.generate_doc_id(source)
                
                doc = Document(
                    content=raw_doc['content'],
                    source=source,
                    doc_id=doc_id,
                    source_type=raw_doc['source_type'],
                    metadata=base_metadata,
                    chunk_id=None,
                    chunk_metadata=None
                )
                documents.append(doc)
        
        return documents
    
    def _extract_web_scrapy_batch(self, urls: List[str], follow_links: bool,
                                   max_pages: int) -> List[Dict]:
        """Extract from URLs using Scrapy"""
        global SCRAPY_RESULTS
        SCRAPY_RESULTS = []
        
        process = CrawlerProcess(settings={
            'LOG_LEVEL': 'ERROR',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': self.config.concurrent_requests,
            'DOWNLOAD_DELAY': self.config.download_delay,
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'DOWNLOAD_TIMEOUT': self.config.download_timeout,
            'RETRY_TIMES': self.config.retry_times,
            'HTTPERROR_ALLOW_ALL': False,
        })
        
        process.crawl(TextSpider, urls=urls, follow_links=follow_links, max_pages=max_pages)
        process.start()
        
        return SCRAPY_RESULTS.copy()
    
    def _extract_web_bs4(self, url: str) -> List[Dict]:
        """Extract from URL using BeautifulSoup (fallback)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=self.config.download_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract title
            title = soup.find('title')
            title = title.text.strip() if title else 'No title'
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_desc_content = ''
            if meta_desc:
                content = meta_desc.get('content', '')
                if content:
                    meta_desc_content = str(content).strip()
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"   ✓ {url[:60]}... ({len(text):,} chars)")
            
            return [{
                'content': text,
                'source': url,
                'source_type': 'web',
                'metadata': {
                    'title': title,
                    'meta_description': meta_desc_content,
                    'url': url,
                    'domain': urlparse(url).netloc,
                    'extracted_at': datetime.now().isoformat(),
                    'char_count': len(text),
                    'word_count': len(text.split()),
                    'extraction_method': 'beautifulsoup'
                }
            }]
        except Exception as e:
            logger.error(f"   ❌ Error: {str(e)[:50]}")
            return []
    
    def _extract_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF"""
        with pdfplumber.open(file_path) as pdf:
            full_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
            
            combined_text = '\n\n'.join(full_text)
            
            return [{
                'content': combined_text,
                'source': file_path,
                'source_type': 'pdf',
                'metadata': {
                    'filename': Path(file_path).name,
                    'total_pages': len(pdf.pages),
                    'extracted_at': datetime.now().isoformat(),
                    'char_count': len(combined_text),
                    'word_count': len(combined_text.split()),
                    'file_size': Path(file_path).stat().st_size
                }
            }]
    
    def _extract_excel(self, file_path: str) -> List[Dict]:
        """Extract text from Excel"""
        documents = []
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Convert to text representation
            text_lines = [' | '.join(str(col) for col in df.columns)]
            for _, row in df.iterrows():
                text_lines.append(' | '.join(str(val) for val in row.values))
            
            text = '\n'.join(text_lines)
            
            documents.append({
                'content': text,
                'source': f"{file_path}::{sheet_name}",
                'source_type': 'excel',
                'metadata': {
                    'filename': Path(file_path).name,
                    'sheet_name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'extracted_at': datetime.now().isoformat(),
                    'file_size': Path(file_path).stat().st_size
                }
            })
        
        return documents
    
    def _extract_csv(self, file_path: str) -> List[Dict]:
        """Extract text from CSV"""
        df = pd.read_csv(file_path)
        
        text_lines = [' | '.join(str(col) for col in df.columns)]
        for _, row in df.iterrows():
            text_lines.append(' | '.join(str(val) for val in row.values))
        
        text = '\n'.join(text_lines)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'csv',
            'metadata': {
                'filename': Path(file_path).name,
                'rows': len(df),
                'columns': len(df.columns),
                'extracted_at': datetime.now().isoformat(),
                'file_size': Path(file_path).stat().st_size
            }
        }]
    
    def _extract_word(self, file_path: str) -> List[Dict]:
        """Extract text from Word document"""
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        # Extract tables
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                tables_text.append(' | '.join(cell.text for cell in row.cells))
        
        text = '\n\n'.join(paragraphs)
        if tables_text:
            text += '\n\n' + '\n'.join(tables_text)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'word',
            'metadata': {
                'filename': Path(file_path).name,
                'paragraphs': len(paragraphs),
                'tables': len(doc.tables),
                'extracted_at': datetime.now().isoformat(),
                'file_size': Path(file_path).stat().st_size
            }
        }]
    
    def _extract_pptx(self, file_path: str) -> List[Dict]:
        """Extract text from PowerPoint"""
        prs = Presentation(file_path)
        all_text = []
        
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_content = getattr(shape, "text", "")
                    if text_content and text_content.strip():
                        slide_text.append(text_content)
            
            if slide_text:
                all_text.append(f"Slide {slide_num}:\n" + '\n'.join(slide_text))
        
        text = '\n\n'.join(all_text)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'powerpoint',
            'metadata': {
                'filename': Path(file_path).name,
                'total_slides': len(prs.slides),
                'extracted_at': datetime.now().isoformat(),
                'file_size': Path(file_path).stat().st_size
            }
        }]
    
    def _extract_txt(self, file_path: str) -> List[Dict]:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'text',
            'metadata': {
                'filename': Path(file_path).name,
                'extracted_at': datetime.now().isoformat(),
                'file_size': Path(file_path).stat().st_size
            }
        }]
    
    def _extract_json(self, file_path: str) -> List[Dict]:
        """Extract text from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = json.dumps(data, indent=2)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'json',
            'metadata': {
                'filename': Path(file_path).name,
                'extracted_at': datetime.now().isoformat(),
                'file_size': Path(file_path).stat().st_size
            }
        }]
    
    def save_documents(self, documents: List[Document], create_index: bool = True):
        """
        Save documents grouped by source as JSON files
        
        Args:
            documents: List of documents to save
            create_index: Create index file with summary
        """
        if not documents:
            logger.warning("⚠️  No documents to save")
            return
        
        logger.info(f"\n💾 Saving to '{self.config.output_dir}/'")
        
        # Group by source
        docs_by_source = {}
        for doc in documents:
            if doc.source not in docs_by_source:
                docs_by_source[doc.source] = []
            docs_by_source[doc.source].append(doc)
        
        # Save each source
        saved_files = []
        for source, docs in docs_by_source.items():
            filename = self._generate_filename(source, docs[0].source_type)
            filepath = os.path.join(self.config.output_dir, filename)
            
            file_data = {
                'source': source,
                'source_type': docs[0].source_type,
                'chunk_method': self.chunker.config.method,
                'total_chunks': len(docs),
                'extracted_at': datetime.now().isoformat(),
                'overall_metadata': {
                    'total_char_count': sum(len(doc.content) for doc in docs),
                    'total_word_count': sum(len(doc.content.split()) for doc in docs),
                    'avg_chunk_size': sum(len(doc.content.split()) for doc in docs) / len(docs)
                },
                'chunks': [doc.to_dict() for doc in docs]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)
            
            saved_files.append(filename)
            logger.info(f"   ✓ {filename} ({len(docs)} chunks)")
        
        logger.info(f"\n✅ Saved {len(saved_files)} file(s)")
        
        if create_index:
            self._create_index_file(docs_by_source)
    
    def _generate_filename(self, source: str, source_type: str) -> str:
        """Generate clean filename from source"""
        if URLValidator.is_url(source):
            parsed = urlparse(source)
            domain = parsed.netloc.replace('www.', '')
            path = parsed.path.strip('/').replace('/', '_')[:50]
            base_name = f"{domain}_{path}" if path else domain
        else:
            base_name = Path(source).stem
            if '::' in base_name:
                base_name = base_name.replace('::', '_')
        
        # Clean filename
        base_name = re.sub(r'[^\w\-_.]', '_', base_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{source_type}_{base_name}_{timestamp}.json"
    
    def _create_index_file(self, docs_by_source: Dict):
        """Create index file with extraction summary"""
        index_data = {
            'chunk_method': self.chunker.config.method,
            'chunk_size': self.chunker.config.target_words,
            'chunk_overlap': self.chunker.config.overlap_words,
            'total_sources': len(docs_by_source),
            'total_chunks': sum(len(docs) for docs in docs_by_source.values()),
            'created_at': datetime.now().isoformat(),
            'sources': []
        }
        
        for source, docs in docs_by_source.items():
            index_data['sources'].append({
                'source': source,
                'source_type': docs[0].source_type,
                'total_chunks': len(docs),
                'total_chars': sum(len(doc.content) for doc in docs),
                'total_words': sum(len(doc.content.split()) for doc in docs),
                'avg_chunk_size': sum(len(doc.content.split()) for doc in docs) / len(docs),
                'metadata': docs[0].metadata
            })
        
        index_path = os.path.join(self.config.output_dir, '_index.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   ✓ _index.json (summary)")


if __name__ == '__main__':
    # Demo
    extractor = DocumentExtractor()
    
    # Test with sample URLs
    sources = [
        "https://en.wikipedia.org/wiki/Embedding_(machine_learning)",
        "https://www.ibm.com/think/topics/embedding"
    ]
    
    logger.info("="*70)
    logger.info("Document Extractor - Demo")
    logger.info("="*70)
    
    docs = extractor.extract_multiple(sources, chunk=True, use_scrapy=False)
    extractor.save_documents(docs)
    
    logger.info(f"\n✅ Demo complete: {len(docs)} documents extracted")