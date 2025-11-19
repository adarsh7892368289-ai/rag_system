import os
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config.settings import EXTRACTION
from core.chunking import EnsembleChunker


class DocumentExtractor:
    
    def __init__(self):
        self.config = EXTRACTION
        self.save_dir = self.config.output_dir
        self.chunker = EnsembleChunker()
        os.makedirs(self.save_dir, exist_ok=True)
    
    def extract_multiple(self, sources: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        all_strategy_documents = {
            'sentence_aware': [],
            'semantic': [],
            'paragraph': [],
            'fixed_size': []
        }
        
        print(f"\n📥 Processing {len(sources)} source(s) in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_source = {}
            
            for source in sources:
                if source.startswith(('http://', 'https://')):
                    future = executor.submit(self._extract_web, source)
                else:
                    future = executor.submit(self._extract_file, source)
                future_to_source[future] = source
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    strategy_docs = future.result()
                    for strategy, docs in strategy_docs.items():
                        all_strategy_documents[strategy].extend(docs)
                    total = sum(len(docs) for docs in strategy_docs.values())
                    source_name = Path(source).name if not source.startswith('http') else urlparse(source).netloc
                    print(f"   ✓ {source_name}: {total} chunks")
                except Exception as e:
                    source_name = Path(source).name if not source.startswith('http') else source
                    print(f"   ✗ {source_name}: {str(e)}")
        
        total_docs = sum(len(docs) for docs in all_strategy_documents.values())
        print(f"\n✅ Extraction complete: {total_docs} total documents across 4 strategies")
        for strategy, docs in all_strategy_documents.items():
            print(f"   • {strategy}: {len(docs)} chunks")
        
        return all_strategy_documents
    
    def _extract_web(self, url: str) -> Dict[str, List[Dict[str, Any]]]:
        headers = {'User-Agent': self.config.user_agent}
        
        response = requests.get(url, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        title = soup.find('title')
        title_text = title.get_text().strip() if title else url
        
        content = soup.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())
        
        if not content or len(content) < 100:
            raise ValueError("Insufficient content extracted")
        
        base_metadata = {
            'source': url,
            'source_type': 'web',
            'title': title_text,
            'url': url,
            'domain': urlparse(url).netloc,
            'extracted_at': datetime.now().isoformat()
        }
        
        return self._chunk_with_all_strategies(content, base_metadata)
    
    def _extract_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
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
            raise ValueError(f"Unsupported file type: {ext}")
        
        content = extractors[ext](path)
        
        base_metadata = {
            'source': str(path),
            'source_type': ext[1:],
            'title': path.stem,
            'filename': path.name,
            'file_type': ext,
            'file_size': path.stat().st_size,
            'extracted_at': datetime.now().isoformat()
        }
        
        return self._chunk_with_all_strategies(content, base_metadata)
    
    def _extract_pdf(self, path: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                return '\n\n'.join(pages)
        except ImportError:
            raise ImportError("pdfplumber required: pip install pdfplumber")
    
    def _extract_word(self, path: Path) -> str:
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs)
        except ImportError:
            raise ImportError("python-docx required: pip install python-docx")
    
    def _extract_excel(self, path: Path) -> str:
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(path)
            all_text = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                lines = [f"Sheet: {sheet_name}"]
                lines.append(' | '.join(str(col) for col in df.columns))
                lines.extend([' | '.join(str(val) for val in row.values) 
                             for _, row in df.iterrows()])
                all_text.append('\n'.join(lines))
            
            return '\n\n'.join(all_text)
        except ImportError:
            raise ImportError("pandas and openpyxl required: pip install pandas openpyxl")
    
    def _extract_csv(self, path: Path) -> str:
        try:
            import pandas as pd
            df = pd.read_csv(path)
            lines = [' | '.join(str(col) for col in df.columns)]
            lines.extend([' | '.join(str(val) for val in row.values) 
                         for _, row in df.iterrows()])
            return '\n'.join(lines)
        except ImportError:
            raise ImportError("pandas required: pip install pandas")
    
    def _extract_pptx(self, path: Path) -> str:
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
            raise ImportError("python-pptx required: pip install python-pptx")
    
    def _extract_txt(self, path: Path) -> str:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_json(self, path: Path) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    def _chunk_with_all_strategies(self, text: str, base_metadata: Dict) -> Dict[str, List[Dict[str, Any]]]:
        strategy_chunks = self.chunker.chunk_all_strategies(text)
        
        strategy_documents = {}
        
        for strategy, chunks in strategy_chunks.items():
            documents = []
            for chunk_obj in chunks:
                doc_id = f"{self._generate_id(base_metadata['source'])}_{strategy}_{chunk_obj.index}"
                
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
            
            strategy_documents[strategy] = documents
        
        return strategy_documents
    
    def save_documents(self, strategy_documents: Dict[str, List[Dict[str, Any]]]):
        if not any(strategy_documents.values()):
            return
        
        print(f"\n💾 Saving to '{self.save_dir}'")
        
        for strategy, documents in strategy_documents.items():
            if not documents:
                continue
            
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
                filename = f"{source_type}_{base_name}_{strategy}_{timestamp}_{source_id}.json"
                filepath = os.path.join(self.save_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source': source,
                        'source_type': source_type,
                        'chunking_strategy': strategy,
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
    
    def load_from_folder(self, folder_path: str) -> Dict[str, List[Dict[str, Any]]]:
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"⚠️  Folder not found: {folder_path}")
            return {}
        
        json_files = list(folder.glob('*.json'))
        
        if not json_files:
            print(f"⚠️  No JSON files in: {folder_path}")
            return {}
        
        all_strategy_documents = {
            'sentence_aware': [],
            'semantic': [],
            'paragraph': [],
            'fixed_size': []
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                strategy = data.get('chunking_strategy', 'sentence_aware')
                
                for i, chunk_data in enumerate(data.get('chunks', [])):
                    doc_id = f"{self._generate_id(data['source'])}_{strategy}_{i}"
                    all_strategy_documents[strategy].append({
                        'id': doc_id,
                        'content': chunk_data['content'],
                        'metadata': chunk_data['metadata']
                    })
                
            except Exception as e:
                print(f"⚠️  Error loading {json_file.name}: {e}")
        
        return all_strategy_documents
    
    def _generate_id(self, source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:12]