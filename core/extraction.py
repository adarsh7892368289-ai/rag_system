import os
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config.settings import EXTRACTION, CHUNKING
from core.chunking import EnsembleChunker, DocumentChunker


class DocumentExtractor:
    """Extract and process documents from various sources"""
    
    def __init__(self, use_ensemble: bool = True):
        self.config = EXTRACTION
        self.save_dir = self.config.output_dir
        self.use_ensemble = use_ensemble 
        
        if use_ensemble:
            self.chunker = EnsembleChunker()
        else:
            self.chunker = DocumentChunker()
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def extract_multiple(self, sources: List[str], chunk: bool = True, 
                        parallel: bool = True, max_chunks_per_doc: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract from multiple sources
        
        Args:
            sources: List of URLs or file paths
            chunk: Whether to chunk documents
            parallel: Process files in parallel
            max_chunks_per_doc: Max chunks per document (default: from config)
        """
        if max_chunks_per_doc is None:
            max_chunks_per_doc = CHUNKING.max_chunks_per_document
        
        web_sources = [s for s in sources if s.startswith(('http://', 'https://'))]
        file_sources = [s for s in sources if not s.startswith(('http://', 'https://'))]
        
        all_documents = []
        
        # Process web sources
        if web_sources:
            print(f"\n🕷️  Processing {len(web_sources)} URL(s)...")
            for url in web_sources:
                try:
                    docs = self._extract_web(url, chunk, max_chunks_per_doc)
                    if docs:
                        all_documents.extend(docs)
                        print(f"   ✓ {url}: {len(docs)} chunks")
                    else:
                        print(f"   ✗ {url}: No content extracted")
                except Exception as e:
                    print(f"   ✗ {url}: {str(e)}")
        
        # Process file sources
        if file_sources:
            if parallel and len(file_sources) > 1:
                all_documents.extend(self._process_files_parallel(file_sources, chunk, max_chunks_per_doc))
            else:
                all_documents.extend(self._process_files_sequential(file_sources, chunk, max_chunks_per_doc))
        
        print(f"\n✅ Extraction complete: {len(all_documents)} documents")
        return all_documents
    
    def _process_files_sequential(self, files: List[str], chunk: bool, max_chunks: int) -> List[Dict[str, Any]]:
        """Process files sequentially"""
        print(f"\n📁 Processing {len(files)} file(s)...")
        all_docs = []
        
        for idx, file_path in enumerate(files, 1):
            try:
                docs = self._extract_file(file_path, chunk, max_chunks)
                if docs:
                    all_docs.extend(docs)
                    print(f"   ✓ [{idx}/{len(files)}] {Path(file_path).name}: {len(docs)} chunks")
            except Exception as e:
                print(f"   ✗ {Path(file_path).name}: {str(e)}")
        
        return all_docs
    
    def _process_files_parallel(self, files: List[str], chunk: bool, max_chunks: int) -> List[Dict[str, Any]]:
        """Process files in parallel"""
        print(f"\n📁 Parallel processing {len(files)} file(s)...")
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_file = {
                executor.submit(self._extract_file, f, chunk, max_chunks): f
                for f in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
                        print(f"   ✓ {Path(file_path).name}: {len(docs)} chunks")
                except Exception as e:
                    print(f"   ✗ {Path(file_path).name}: {str(e)}")
        
        return all_docs
    
    def _extract_web(self, url: str, chunk: bool = True, 
                     max_chunks: int = None) -> List[Dict[str, Any]]:
        """Extract content from web URL"""
        headers = {'User-Agent': self.config.user_agent}
        
        try:
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"{e.__class__.__name__}: {str(e)}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else url
        
        content = soup.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())
        
        if not content or len(content) < 100:
            return []
        
        base_metadata = {
            'source': url,
            'source_type': 'web',
            'title': title_text,
            'url': url,
            'domain': urlparse(url).netloc,
            'extracted_at': datetime.now().isoformat()
        }
        
        if not chunk:
            doc_id = self._generate_id(url)
            return [{
                'id': doc_id,
                'content': content,
                'metadata': base_metadata
            }]
        
        return self._chunk_and_create_documents(content, base_metadata, max_chunks)
    
    def _extract_file(self, file_path: str, chunk: bool = True,
                      max_chunks: int = None) -> List[Dict[str, Any]]:
        """Extract content from file (PDF, DOCX, XLSX, CSV, PPTX, TXT, JSON)"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        # Map file extensions to extraction methods
        extractors = {
            '.pdf': self._extract_pdf,
            '.docx': self._extract_word,
            '.doc': self._extract_word,
            '.xlsx': self._extract_excel,
            '.xls': self._extract_excel,
            '.csv': self._extract_csv,
            '.pptx': self._extract_pptx,
            '.txt': self._extract_txt,
            '.json': self._extract_json,
        }
        
        if ext not in extractors:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(extractors.keys())}")
        
        # Extract content using appropriate method
        content = extractors[ext](path)
        
        # Build base metadata
        base_metadata = {
            'source': str(path),
            'source_type': ext[1:],  # Remove leading dot
            'title': path.stem,
            'filename': path.name,
            'file_type': ext,
            'file_size': path.stat().st_size,
            'extracted_at': datetime.now().isoformat()
        }
        
        if not chunk:
            doc_id = self._generate_id(str(path))
            return [{
                'id': doc_id,
                'content': content,
                'metadata': base_metadata
            }]
        
        return self._chunk_and_create_documents(content, base_metadata, max_chunks)
    
    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF"""
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                return '\n\n'.join(pages)
        except ImportError:
            raise ImportError("pdfplumber required for PDF files: pip install pdfplumber")
    
    def _extract_word(self, path: Path) -> str:
        """Extract text from Word document"""
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs)
        except ImportError:
            raise ImportError("python-docx required for Word files: pip install python-docx")
    
    def _extract_excel(self, path: Path) -> str:
        """Extract data from Excel file"""
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(path)
            all_text = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Format sheet data
                lines = [f"Sheet: {sheet_name}"]
                lines.append(' | '.join(str(col) for col in df.columns))
                lines.extend([' | '.join(str(val) for val in row.values) 
                             for _, row in df.iterrows()])
                
                all_text.append('\n'.join(lines))
            
            return '\n\n'.join(all_text)
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel files: pip install pandas openpyxl")
    
    def _extract_csv(self, path: Path) -> str:
        """Extract data from CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            
            lines = [' | '.join(str(col) for col in df.columns)]
            lines.extend([' | '.join(str(val) for val in row.values) 
                         for _, row in df.iterrows()])
            
            return '\n'.join(lines)
        except ImportError:
            raise ImportError("pandas required for CSV files: pip install pandas")
    
    def _extract_pptx(self, path: Path) -> str:
        """Extract text from PowerPoint presentation"""
        try:
            from pptx import Presentation
            prs = Presentation(path)
            all_text = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, 'text') and shape.text.strip():
                        slide_text.append(shape.text.strip())
                
                if slide_text:
                    all_text.append(f"Slide {slide_num}:\n" + '\n'.join(slide_text))
            
            return '\n\n'.join(all_text)
        except ImportError:
            raise ImportError("python-pptx required for PowerPoint files: pip install python-pptx")
    
    def _extract_txt(self, path: Path) -> str:
        """Extract text from plain text file"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_json(self, path: Path) -> str:
        """Extract content from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    def _chunk_and_create_documents(self, text: str, base_metadata: Dict,
                                    max_chunks: int = None) -> List[Dict[str, Any]]:
        """
        Chunk text and create document objects
        
        Args:
            text: Text to chunk
            base_metadata: Base metadata for all chunks
            max_chunks: Max chunks to keep (uses config default if None)
        """
        if max_chunks is None:
            max_chunks = CHUNKING.max_chunks_per_document
        
        if self.use_ensemble:
            chunks = self.chunker.chunk_with_ensemble(text, max_chunks=max_chunks)
        else:
            chunks = self.chunker.chunk(text, filter_quality=True)
            if max_chunks and len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
        
        documents = []
        for chunk_obj in chunks:
            doc_id = f"{self._generate_id(base_metadata['source'])}_{chunk_obj.index}"
            
            chunk_metadata = {
                **base_metadata,
                'chunk_index': chunk_obj.index,
                'chunk_word_count': chunk_obj.word_count,
                'chunk_char_count': chunk_obj.char_count,
                **chunk_obj.metadata
            }
            
            documents.append({
                'id': doc_id,
                'content': chunk_obj.text,
                'metadata': chunk_metadata
            })
        
        return documents
    
    def save_documents(self, documents: List[Dict[str, Any]]):
        """Save extracted documents to JSON"""
        if not documents:
            return
        
        print(f"\n💾 Saving to '{self.save_dir}'")
        
        by_source = {}
        for doc in documents:
            source = doc['metadata']['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(doc)
        
        for source, docs in by_source.items():
            source_type = docs[0]['metadata']['source_type']
            source_id = self._generate_id(source)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create unique filename
            if source.startswith(('http://', 'https://')):
                parsed = urlparse(source)
                domain = parsed.netloc.replace('www.', '')
                path_parts = [p for p in parsed.path.split('/') if p]
                page_id = path_parts[-1] if path_parts else 'index'
                page_id = page_id.replace('.html', '').replace('.php', '')[:30]
                base_name = f"{domain}_{page_id}" if page_id != 'index' else domain
            else:
                base_name = Path(source).stem
            
            base_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in base_name)
            filename = f"{source_type}_{base_name}_{timestamp}_{source_id}.json"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'source': source,
                    'source_type': source_type,
                    'total_chunks': len(docs),
                    'extracted_at': docs[0]['metadata']['extracted_at'],
                    'chunks': [
                        {
                            'content': doc['content'],
                            'metadata': doc['metadata']
                        }
                        for doc in docs
                    ]
                }, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ {filename} ({len(docs)} chunks)")
    
    def load_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Load documents from saved JSON files"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder_path}")
            return []
        
        json_files = list(folder.glob('*.json'))
        
        if not json_files:
            print(f"⚠️  No JSON files in: {folder_path}")
            return []
        
        all_documents = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for i, chunk_data in enumerate(data.get('chunks', [])):
                    doc_id = f"{self._generate_id(data['source'])}_{i}"
                    all_documents.append({
                        'id': doc_id,
                        'content': chunk_data['content'],
                        'metadata': chunk_data['metadata']
                    })
                
            except Exception as e:
                print(f"⚠️  Error loading {json_file.name}: {e}")
        
        return all_documents
    
    def _generate_id(self, source: str) -> str:
        """Generate unique ID from source"""
        return hashlib.md5(source.encode()).hexdigest()[:12]