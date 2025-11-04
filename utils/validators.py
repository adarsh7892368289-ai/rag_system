"""
Input Validation and Error Handling

Validates user inputs, file paths, URLs, and configuration.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Union, Tuple
from urllib.parse import urlparse
from utils.logger import get_logger

logger = get_logger("validators")


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class FileValidator:
    """Validate files and file paths"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF Document',
        '.xlsx': 'Excel Spreadsheet',
        '.xls': 'Excel Spreadsheet (Legacy)',
        '.csv': 'CSV File',
        '.docx': 'Word Document',
        '.pptx': 'PowerPoint Presentation',
        '.txt': 'Text File',
        '.json': 'JSON File'
    }
    
    @staticmethod
    def validate_file_exists(file_path: str) -> Path:
        """
        Validate that file exists
        
        Args:
            file_path: Path to file
        
        Returns:
            Path object
        
        Raises:
            ValidationError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        return path
    
    @staticmethod
    def validate_file_extension(file_path: str) -> Tuple[Path, str]:
        """
        Validate that file has supported extension
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (Path object, extension)
        
        Raises:
            ValidationError: If extension not supported
        """
        path = FileValidator.validate_file_exists(file_path)
        ext = path.suffix.lower()
        
        if ext not in FileValidator.SUPPORTED_EXTENSIONS:
            supported = ", ".join(FileValidator.SUPPORTED_EXTENSIONS.keys())
            raise ValidationError(
                f"Unsupported file type: {ext}\n"
                f"Supported types: {supported}"
            )
        
        return path, ext
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: int = 100) -> int:
        """
        Validate file size
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum allowed size in MB
        
        Returns:
            File size in bytes
        
        Raises:
            ValidationError: If file too large
        """
        path = FileValidator.validate_file_exists(file_path)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)\n"
                f"File: {file_path}"
            )
        
        return size_bytes
    
    @staticmethod
    def validate_directory(dir_path: str, create: bool = False) -> Path:
        """
        Validate directory exists or create it
        
        Args:
            dir_path: Path to directory
            create: Create if doesn't exist
        
        Returns:
            Path object
        
        Raises:
            ValidationError: If directory invalid
        """
        path = Path(dir_path)
        
        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                raise ValidationError(f"Directory not found: {dir_path}")
        
        if not path.is_dir():
            raise ValidationError(f"Path is not a directory: {dir_path}")
        
        return path


class URLValidator:
    """Validate URLs and web resources"""
    
    VALID_SCHEMES = ['http', 'https']
    
    @staticmethod
    def validate_url(url: str) -> str:
        """
        Validate URL format
        
        Args:
            url: URL string
        
        Returns:
            Cleaned URL
        
        Raises:
            ValidationError: If URL invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {url}\nError: {e}")
        
        # Check scheme
        if parsed.scheme not in URLValidator.VALID_SCHEMES:
            raise ValidationError(
                f"Invalid URL scheme: {parsed.scheme}\n"
                f"Supported: {', '.join(URLValidator.VALID_SCHEMES)}"
            )
        
        # Check netloc (domain)
        if not parsed.netloc:
            raise ValidationError(f"URL missing domain: {url}")
        
        return url
    
    @staticmethod
    def validate_url_list(urls: List[str]) -> List[str]:
        """
        Validate list of URLs
        
        Args:
            urls: List of URL strings
        
        Returns:
            List of valid URLs
        
        Raises:
            ValidationError: If any URL invalid
        """
        if not urls:
            raise ValidationError("URL list is empty")
        
        if not isinstance(urls, list):
            raise ValidationError("URLs must be provided as a list")
        
        valid_urls = []
        errors = []
        
        for i, url in enumerate(urls, 1):
            try:
                valid_url = URLValidator.validate_url(url)
                valid_urls.append(valid_url)
            except ValidationError as e:
                errors.append(f"URL {i}: {e}")
        
        if errors:
            raise ValidationError(f"Invalid URLs:\n" + "\n".join(errors))
        
        return valid_urls
    
    @staticmethod
    def is_url(source: str) -> bool:
        """Check if string is a URL"""
        return source.startswith(('http://', 'https://'))


class TextValidator:
    """Validate text inputs"""
    
    @staticmethod
    def validate_text(text: str, 
                     min_length: int = 1, 
                     max_length: Optional[int] = None,
                     field_name: str = "Text") -> str:
        """
        Validate text input
        
        Args:
            text: Text to validate
            min_length: Minimum length
            max_length: Maximum length (None for no limit)
            field_name: Name of field for error messages
        
        Returns:
            Cleaned text
        
        Raises:
            ValidationError: If text invalid
        """
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string")
        
        # Strip whitespace
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(
                f"{field_name} too short: {len(text)} chars (min: {min_length})"
            )
        
        if max_length and len(text) > max_length:
            raise ValidationError(
                f"{field_name} too long: {len(text)} chars (max: {max_length})"
            )
        
        return text
    
    @staticmethod
    def validate_query(query: str) -> str:
        """
        Validate search query
        
        Args:
            query: Search query
        
        Returns:
            Cleaned query
        
        Raises:
            ValidationError: If query invalid
        """
        query = TextValidator.validate_text(
            query,
            min_length=2,
            max_length=500,
            field_name="Query"
        )
        
        # Check if meaningful (not just punctuation)
        if not re.search(r'[a-zA-Z0-9]', query):
            raise ValidationError("Query must contain alphanumeric characters")
        
        return query


class ConfigValidator:
    """Validate configuration parameters"""
    
    @staticmethod
    def validate_chunk_config(target_words: int, 
                             overlap_words: int,
                             min_chunk_words: int) -> None:
        """
        Validate chunking configuration
        
        Raises:
            ValidationError: If configuration invalid
        """
        if target_words < 50:
            raise ValidationError(f"target_words too small: {target_words} (min: 50)")
        
        if target_words > 1000:
            raise ValidationError(f"target_words too large: {target_words} (max: 1000)")
        
        if overlap_words < 0:
            raise ValidationError(f"overlap_words must be positive: {overlap_words}")
        
        if overlap_words >= target_words:
            raise ValidationError(
                f"overlap_words ({overlap_words}) must be less than target_words ({target_words})"
            )
        
        if min_chunk_words < 10:
            raise ValidationError(f"min_chunk_words too small: {min_chunk_words} (min: 10)")
        
        if min_chunk_words >= target_words:
            raise ValidationError(
                f"min_chunk_words ({min_chunk_words}) must be less than target_words ({target_words})"
            )
    
    @staticmethod
    def validate_search_params(top_k: int, 
                               alpha: Optional[float] = None,
                               lambda_param: Optional[float] = None) -> None:
        """
        Validate search parameters
        
        Raises:
            ValidationError: If parameters invalid
        """
        if top_k < 1:
            raise ValidationError(f"top_k must be at least 1: {top_k}")
        
        if top_k > 100:
            raise ValidationError(f"top_k too large: {top_k} (max: 100)")
        
        if alpha is not None:
            if not 0.0 <= alpha <= 1.0:
                raise ValidationError(f"alpha must be in [0.0, 1.0]: {alpha}")
        
        if lambda_param is not None:
            if not 0.0 <= lambda_param <= 1.0:
                raise ValidationError(f"lambda_param must be in [0.0, 1.0]: {lambda_param}")


class SourceValidator:
    """Validate sources (URLs or files)"""
    
    @staticmethod
    def validate_source(source: str) -> Tuple[str, str]:
        """
        Validate source and determine type
        
        Args:
            source: URL or file path
        
        Returns:
            Tuple of (source, type) where type is 'url' or 'file'
        
        Raises:
            ValidationError: If source invalid
        """
        if URLValidator.is_url(source):
            URLValidator.validate_url(source)
            return (source, 'url')
        else:
            FileValidator.validate_file_extension(source)
            return (source, 'file')
    
    @staticmethod
    def validate_sources(sources: List[str]) -> List[Tuple[str, str]]:
        """
        Validate multiple sources
        
        Args:
            sources: List of URLs or file paths
        
        Returns:
            List of (source, type) tuples
        
        Raises:
            ValidationError: If any source invalid
        """
        if not sources:
            raise ValidationError("Source list is empty")
        
        if not isinstance(sources, list):
            raise ValidationError("Sources must be provided as a list")
        
        validated = []
        errors = []
        
        for i, source in enumerate(sources, 1):
            try:
                validated.append(SourceValidator.validate_source(source))
            except ValidationError as e:
                errors.append(f"Source {i} ({source}): {e}")
        
        if errors:
            raise ValidationError(f"Invalid sources:\n" + "\n".join(errors))
        
        return validated


# Convenience functions
def validate_and_log(validator_func, *args, operation: str = "Validation", **kwargs):
    """
    Run validation and log result
    
    Args:
        validator_func: Validation function to call
        *args: Positional arguments for validator
        operation: Operation name for logging
        **kwargs: Keyword arguments for validator
    
    Returns:
        Validation result
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        result = validator_func(*args, **kwargs)
        logger.debug(f"✅ {operation} passed")
        return result
    except ValidationError as e:
        logger.error(f"❌ {operation} failed: {e}")
        raise


if __name__ == "__main__":
    # Demo validation
    print("Testing validators...\n")
    
    # File validation
    try:
        FileValidator.validate_file_extension("test.pdf")
    except ValidationError as e:
        print(f"Expected error: {e}\n")
    
    # URL validation
    try:
        url = URLValidator.validate_url("https://example.com")
        print(f"Valid URL: {url}\n")
    except ValidationError as e:
        print(f"Error: {e}\n")
    
    # Text validation
    try:
        query = TextValidator.validate_query("What is machine learning?")
        print(f"Valid query: {query}\n")
    except ValidationError as e:
        print(f"Error: {e}\n")
    
    # Config validation
    try:
        ConfigValidator.validate_chunk_config(150, 30, 50)
        print("Valid chunk config\n")
    except ValidationError as e:
        print(f"Error: {e}\n")