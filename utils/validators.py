"""
Input Validators - Clean Production Version

Validates user inputs, configurations, and data.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class TextValidator:
    """Validate text inputs"""
    
    @staticmethod
    def validate_text(text: str, 
                     min_length: int = 1,
                     max_length: Optional[int] = None,
                     field_name: str = "Text") -> str:
        """
        Validate text content
        
        Args:
            text: Text to validate
            min_length: Minimum length
            max_length: Maximum length (None for unlimited)
            field_name: Name for error messages
        
        Returns:
            Validated text (stripped)
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string")
        
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters"
            )
        
        if max_length and len(text) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters"
            )
        
        return text
    
    @staticmethod
    def validate_query(query: str) -> str:
        """
        Validate search query
        
        Args:
            query: Search query
        
        Returns:
            Validated query
        
        Raises:
            ValidationError: If query is invalid
        """
        query = TextValidator.validate_text(
            query,
            min_length=1,
            max_length=1000,
            field_name="Search query"
        )
        
        # Check for suspicious patterns
        if len(query) > 500:
            raise ValidationError("Query too long (max 500 characters)")
        
        return query


class URLValidator:
    """Validate URLs"""
    
    @staticmethod
    def is_url(source: str) -> bool:
        """
        Check if string is a URL
        
        Args:
            source: String to check
        
        Returns:
            True if URL, False otherwise
        """
        return source.startswith(('http://', 'https://'))
    
    @staticmethod
    def validate_url(url: str) -> str:
        """
        Validate URL format
        
        Args:
            url: URL to validate
        
        Returns:
            Validated URL
        
        Raises:
            ValidationError: If URL is invalid
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")
        
        url = url.strip()
        
        if not url:
            raise ValidationError("URL cannot be empty")
        
        if not URLValidator.is_url(url):
            raise ValidationError(
                f"Invalid URL format: {url}. Must start with http:// or https://"
            )
        
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValidationError(f"Invalid URL: missing domain in {url}")
        except Exception as e:
            raise ValidationError(f"Invalid URL: {e}")
        
        return url


class FileValidator:
    """Validate file paths and extensions"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.xlsx', '.xls', '.csv', '.docx', 
        '.pptx', '.txt', '.json'
    }
    
    @staticmethod
    def validate_file_exists(file_path: str) -> Path:
        """
        Validate file exists
        
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
            raise ValidationError(f"Not a file: {file_path}")
        
        return path
    
    @staticmethod
    def validate_file_extension(file_path: str) -> Tuple[Path, str]:
        """
        Validate file extension
        
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
            raise ValidationError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(FileValidator.SUPPORTED_EXTENSIONS))}"
            )
        
        return path, ext


class SourceValidator:
    """Validate sources (URLs or files)"""
    
    @staticmethod
    def validate_source(source: str) -> Tuple[str, str]:
        """
        Validate source (URL or file)
        
        Args:
            source: URL or file path
        
        Returns:
            Tuple of (validated_source, source_type)
            source_type is either 'url' or 'file'
        
        Raises:
            ValidationError: If source is invalid
        """
        if not isinstance(source, str):
            raise ValidationError("Source must be a string")
        
        source = source.strip()
        
        if not source:
            raise ValidationError("Source cannot be empty")
        
        # Check if URL
        if URLValidator.is_url(source):
            validated = URLValidator.validate_url(source)
            return validated, 'url'
        
        # Check if file
        try:
            path, _ = FileValidator.validate_file_extension(source)
            return str(path), 'file'
        except ValidationError as e:
            raise ValidationError(
                f"Invalid source '{source}': not a valid URL or file. {e}"
            )
    
    @staticmethod
    def validate_sources(sources: List[str]) -> List[Tuple[str, str]]:
        """
        Validate multiple sources
        
        Args:
            sources: List of URLs or file paths
        
        Returns:
            List of (validated_source, source_type) tuples
        
        Raises:
            ValidationError: If any source is invalid
        """
        if not isinstance(sources, list):
            raise ValidationError("Sources must be a list")
        
        if not sources:
            raise ValidationError("Sources list cannot be empty")
        
        validated = []
        errors = []
        
        for i, source in enumerate(sources, 1):
            try:
                validated.append(SourceValidator.validate_source(source))
            except ValidationError as e:
                errors.append(f"Source {i}: {e}")
        
        if errors:
            raise ValidationError(
                f"Validation failed for {len(errors)} source(s):\n" + 
                "\n".join(errors)
            )
        
        return validated


class ConfigValidator:
    """Validate configuration parameters"""
    
    @staticmethod
    def validate_positive_int(value: int, 
                             param_name: str,
                             min_value: int = 1,
                             max_value: Optional[int] = None) -> int:
        """
        Validate positive integer
        
        Args:
            value: Value to validate
            param_name: Parameter name for errors
            min_value: Minimum allowed value
            max_value: Maximum allowed value (None for unlimited)
        
        Returns:
            Validated value
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, int):
            raise ValidationError(f"{param_name} must be an integer")
        
        if value < min_value:
            raise ValidationError(
                f"{param_name} must be at least {min_value}"
            )
        
        if max_value and value > max_value:
            raise ValidationError(
                f"{param_name} must be at most {max_value}"
            )
        
        return value
    
    @staticmethod
    def validate_float_range(value: float,
                            param_name: str,
                            min_value: float = 0.0,
                            max_value: float = 1.0) -> float:
        """
        Validate float in range
        
        Args:
            value: Value to validate
            param_name: Parameter name for errors
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        
        Returns:
            Validated value
        
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{param_name} must be a number")
        
        value = float(value)
        
        if value < min_value or value > max_value:
            raise ValidationError(
                f"{param_name} must be between {min_value} and {max_value}"
            )
        
        return value
    
    @staticmethod
    def validate_chunk_config(target_words: int,
                             overlap_words: int,
                             min_chunk_words: int):
        """
        Validate chunking configuration
        
        Args:
            target_words: Target chunk size
            overlap_words: Overlap size
            min_chunk_words: Minimum chunk size
        
        Raises:
            ValidationError: If configuration is invalid
        """
        ConfigValidator.validate_positive_int(
            target_words, 
            "target_words",
            min_value=10,
            max_value=10000
        )
        
        ConfigValidator.validate_positive_int(
            overlap_words,
            "overlap_words",
            min_value=0,
            max_value=target_words
        )
        
        ConfigValidator.validate_positive_int(
            min_chunk_words,
            "min_chunk_words",
            min_value=1,
            max_value=target_words
        )
        
        if overlap_words >= target_words:
            raise ValidationError(
                "overlap_words must be less than target_words"
            )
        
        if min_chunk_words > target_words:
            raise ValidationError(
                "min_chunk_words cannot be greater than target_words"
            )
    
    @staticmethod
    def validate_search_params(n_results: int,
                               alpha: Optional[float] = None,
                               lambda_param: Optional[float] = None):
        """
        Validate search parameters
        
        Args:
            n_results: Number of results to return
            alpha: Hybrid search alpha (optional)
            lambda_param: MMR lambda parameter (optional)
        
        Raises:
            ValidationError: If parameters are invalid
        """
        ConfigValidator.validate_positive_int(
            n_results,
            "n_results",
            min_value=1,
            max_value=100
        )
        
        if alpha is not None:
            ConfigValidator.validate_float_range(
                alpha,
                "alpha",
                min_value=0.0,
                max_value=1.0
            )
        
        if lambda_param is not None:
            ConfigValidator.validate_float_range(
                lambda_param,
                "lambda_param",
                min_value=0.0,
                max_value=1.0
            )


if __name__ == "__main__":
    # Demo
    print("Testing Validators:\n")
    
    # Text validation
    try:
        text = TextValidator.validate_text("   Hello World   ")
        print(f"✅ Valid text: '{text}'")
    except ValidationError as e:
        print(f"❌ {e}")
    
    # URL validation
    try:
        url = URLValidator.validate_url("https://example.com")
        print(f"✅ Valid URL: {url}")
    except ValidationError as e:
        print(f"❌ {e}")
    
    # File validation
    try:
        path, ext = FileValidator.validate_file_extension("test.pdf")
        print(f"✅ Valid file: {path} ({ext})")
    except ValidationError as e:
        print(f"❌ {e}")
    
    # Config validation
    try:
        ConfigValidator.validate_chunk_config(
            target_words=150,
            overlap_words=30,
            min_chunk_words=50
        )
        print("✅ Valid chunk config")
    except ValidationError as e:
        print(f"❌ {e}")
    
    # Search params validation
    try:
        ConfigValidator.validate_search_params(
            n_results=5,
            alpha=0.7,
            lambda_param=0.7
        )
        print("✅ Valid search params")
    except ValidationError as e:
        print(f"❌ {e}")