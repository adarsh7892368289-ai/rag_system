import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation

from core.chunking import DocumentChunker
from config.settings import EXTRACTION

@dataclass
class Document:
    content: str
    source: str
    doc_id: str
    source_type: str
    metadata: Dict
    chunk_id: int
    
    def to_dict(self):
        return asdict(self)

class DocumentExtractor:
    """Extract and process documents from multiple sources"""
    
    def __init__(self):
        self.chunker = DocumentChunker()
        os.makedirs(EXTRACTION.output_dir, exist_ok=True)
    
    def extract_multiple(self, sources: List[str], chunk: bool = True, parallel: bool = True) -> List[Document]:
        """Extract from multiple sources (URLs and files)"""
        urls = [s for s in sources if s.startswith(('http://', 'https://'))]
        files = [s for s in sources if not s.startswith(('http://', 'https://'))]
        
        all_documents = []
        
        if urls:
            print(f"\n🕷️  Processing {len(urls)} URL(s)...")
            for url in urls:
                try:
                    raw_docs = self._extract_web(url)
                    docs = self._convert_to_documents(raw_docs, chunk)
                    all_documents.extend(docs)
                    print(f"   ✓ {url}: {len(docs)} docs")
                except Exception as e:
                    print(f"   ✗ {url}: {e}")
        
        if files:
            if parallel and len(files) > 1:
                all_documents.extend(self._process_files_parallel(files, chunk))
            else:
                all_documents.extend(self._process_files_sequential(files, chunk))
        
        print(f"\n✅ Extraction complete: {len(all_documents)} documents")
        return all_documents
    
    def _process_files_sequential(self, files: List[str], chunk: bool) -> List[Document]:
        """Process files sequentially"""
        print(f"\n📁 Processing {len(files)} file(s)...")
        all_docs = []
        
        for idx, file_path in enumerate(files, 1):
            try:
                raw_docs = self._extract_file(file_path)
                docs = self._convert_to_documents(raw_docs, chunk)
                all_docs.extend(docs)
                print(f"   ✓ [{idx}/{len(files)}] {Path(file_path).name}: {len(docs)} docs")
            except Exception as e:
                print(f"   ✗ {Path(file_path).name}: {e}")
        
        return all_docs
    
    def _process_files_parallel(self, files: List[str], chunk: bool) -> List[Document]:
        """Process files in parallel"""
        print(f"\n📁 Parallel processing {len(files)} file(s)...")
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=EXTRACTION.max_workers) as executor:
            future_to_file = {
                executor.submit(self._extract_and_convert, f, chunk): f
                for f in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    print(f"   ✓ {Path(file_path).name}: {len(docs)} docs")
                except Exception as e:
                    print(f"   ✗ {Path(file_path).name}: {e}")
        
        return all_docs
    
    def _extract_and_convert(self, file_path: str, chunk: bool) -> List[Document]:
        """Extract and convert a single file"""
        raw_docs = self._extract_file(file_path)
        return self._convert_to_documents(raw_docs, chunk)
    
    def _extract_file(self, file_path: str) -> List[Dict]:
        """Extract content from a file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
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
        
        if ext not in extractors:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return extractors[ext](str(path))
    
    def _convert_to_documents(self, raw_docs: List[Dict], do_chunk: bool) -> List[Document]:
        """Convert raw documents to Document objects with chunking"""
        documents = []
        
        for raw_doc in raw_docs:
            source = raw_doc['source']
            doc_id = self._generate_doc_id(source)
            
            if do_chunk:
                chunks = self.chunker.chunk(raw_doc['content'])
                
                for chunk in chunks:
                    chunk_metadata = {
                        'chunk_index': chunk.index,
                        'chunk_word_count': chunk.word_count,
                        'chunk_keywords': chunk.metadata.get('keywords', []),
                        **raw_doc['metadata']
                    }
                    
                    doc = Document(
                        content=chunk.text,
                        source=source,
                        doc_id=f"{doc_id}_chunk{chunk.index}",
                        source_type=raw_doc['source_type'],
                        metadata=chunk_metadata,
                        chunk_id=chunk.index
                    )
                    documents.append(doc)
            else:
                doc = Document(
                    content=raw_doc['content'],
                    source=source,
                    doc_id=doc_id,
                    source_type=raw_doc['source_type'],
                    metadata=raw_doc['metadata'],
                    chunk_id=None
                )
                documents.append(doc)
        
        return documents
    
    def _generate_doc_id(self, source: str) -> str:
        """Generate unique document ID from source"""
        source_hash = hashlib.md5(source.encode()).hexdigest()[:12]
        
        if source.startswith(('http://', 'https://')):
            domain = urlparse(source).netloc.replace('www.', '')
            return f"web_{domain}_{source_hash}"
        else:
            filename = Path(source).stem
            return f"file_{filename}_{source_hash}"
    
    def _extract_web(self, url: str) -> List[Dict]:
        """Extract content from web page"""
        headers = {'User-Agent': EXTRACTION.user_agent}
        response = requests.get(url, headers=headers, timeout=EXTRACTION.timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('title')
        title = title.text.strip() if title else 'No title'
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
        
        return [{
            'content': text,
            'source': url,
            'source_type': 'web',
            'metadata': {
                'title': title,
                'url': url,
                'domain': urlparse(url).netloc,
                'extracted_at': datetime.now().isoformat()
            }
        }]
    
    def _extract_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF"""
        with pdfplumber.open(file_path) as pdf:
            text = '\n\n'.join([p.extract_text() for p in pdf.pages if p.extract_text()])
            
            return [{
                'content': text,
                'source': file_path,
                'source_type': 'pdf',
                'metadata': {
                    'filename': Path(file_path).name,
                    'total_pages': len(pdf.pages),
                    'extracted_at': datetime.now().isoformat()
                }
            }]
    
    def _extract_excel(self, file_path: str) -> List[Dict]:
        """Extract data from Excel file"""
        documents = []
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            text_lines = [' | '.join(str(col) for col in df.columns)]
            text_lines.extend([' | '.join(str(val) for val in row.values) 
                             for _, row in df.iterrows()])
            text = '\n'.join(text_lines)
            
            documents.append({
                'content': text,
                'source': f"{file_path}::{sheet_name}",
                'source_type': 'excel',
                'metadata': {
                    'filename': Path(file_path).name,
                    'sheet_name': sheet_name,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            })
        
        return documents
    
    def _extract_csv(self, file_path: str) -> List[Dict]:
        """Extract data from CSV file"""
        df = pd.read_csv(file_path)
        text_lines = [' | '.join(str(col) for col in df.columns)]
        text_lines.extend([' | '.join(str(val) for val in row.values) 
                         for _, row in df.iterrows()])
        text = '\n'.join(text_lines)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'csv',
            'metadata': {
                'filename': Path(file_path).name,
                'rows': len(df),
                'columns': len(df.columns)
            }
        }]
    
    def _extract_word(self, file_path: str) -> List[Dict]:
        """Extract text from Word document"""
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = '\n\n'.join(paragraphs)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'word',
            'metadata': {
                'filename': Path(file_path).name,
                'paragraphs': len(paragraphs)
            }
        }]
    
    def _extract_pptx(self, file_path: str) -> List[Dict]:
        """Extract text from PowerPoint presentation"""
        prs = Presentation(file_path)
        all_text = []
        
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, 'text') and shape.text.strip():
                    slide_text.append(shape.text.strip())
            
            if slide_text:
                all_text.append(f"Slide {slide_num}:\n" + '\n'.join(slide_text))
        
        text = '\n\n'.join(all_text)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'powerpoint',
            'metadata': {
                'filename': Path(file_path).name,
                'total_slides': len(prs.slides)
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
            'metadata': {'filename': Path(file_path).name}
        }]
    
    def _extract_json(self, file_path: str) -> List[Dict]:
        """Extract content from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = json.dumps(data, indent=2)
        
        return [{
            'content': text,
            'source': file_path,
            'source_type': 'json',
            'metadata': {'filename': Path(file_path).name}
        }]
    
    def save_documents(self, documents: List[Document]):
        """Save extracted documents to JSON files with unique filenames"""
        if not documents:
            return
        
        print(f"\n💾 Saving to '{EXTRACTION.output_dir}/'")
        
        # Group documents by source
        docs_by_source = {}
        for doc in documents:
            if doc.source not in docs_by_source:
                docs_by_source[doc.source] = []
            docs_by_source[doc.source].append(doc)
        
        # Save each source to a unique file
        for source, docs in docs_by_source.items():
            filename = self._generate_unique_filename(source, docs[0].source_type)
            filepath = os.path.join(EXTRACTION.output_dir, filename)
            
            file_data = {
                'source': source,
                'source_type': docs[0].source_type,
                'total_chunks': len(docs),
                'extracted_at': datetime.now().isoformat(),
                'chunks': [doc.to_dict() for doc in docs]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ {filename} ({len(docs)} chunks)")
    
    def _generate_unique_filename(self, source: str, source_type: str) -> str:
        """
        Generate unique filename using source hash to prevent duplicates
        
        Format: {type}_{domain/filename}_{timestamp}_{hash}.json
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create unique hash from full source URL/path
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        
        if source.startswith(('http://', 'https://')):
            # Web source: extract domain and page identifier
            parsed = urlparse(source)
            domain = parsed.netloc.replace('www.', '')
            
            # Get page identifier from path
            path_parts = [p for p in parsed.path.split('/') if p]
            page_id = path_parts[-1] if path_parts else 'index'
            page_id = page_id.replace('.html', '').replace('.php', '')[:30]
            
            base_name = f"{domain}_{page_id}" if page_id != 'index' else domain
        else:
            # File source: use filename
            base_name = Path(source).stem
        
        # Sanitize filename
        base_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in base_name)
        
        # Combine: type_basename_timestamp_hash.json
        filename = f"{source_type}_{base_name}_{timestamp}_{source_hash}.json"
        
        return filename