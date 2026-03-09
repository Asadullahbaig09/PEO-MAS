"""
Async I/O Operations for Perpetual Ethical Oversight MAS

Provides:
- Asynchronous HTTP requests with aiohttp
- Concurrent signal scraping
- Async context managers
- Connection pooling
- Timeout handling
"""

import asyncio
from typing import List, Optional, Dict, Any, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
import logging

# Optional async dependencies
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


@dataclass
class AsyncRequestConfig:
    """Configuration for async HTTP requests"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_concurrent: int = 5
    use_connection_pool: bool = True
    pool_size: int = 10
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class AsyncHTTPClient:
    """
    Asynchronous HTTP client with connection pooling and retry logic
    
    Features:
    - Connection pooling for efficiency
    - Configurable timeout and retries
    - Concurrent request handling
    - Error categorization
    - Automatic retry on failure
    """
    
    def __init__(self, config: Optional[AsyncRequestConfig] = None):
        """
        Initialize async HTTP client
        
        Args:
            config: Request configuration
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for async I/O. Install with: pip install aiohttp")
        
        self.config = config or AsyncRequestConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector: Optional[aiohttp.TCPConnector] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def open(self):
        """Open connection pool and session"""
        if self.config.use_connection_pool:
            self.connector = aiohttp.TCPConnector(
                limit=self.config.pool_size,
                limit_per_host=5,
                ttl_dns_cache=300
            )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers=self.config.headers
        )
    
    async def close(self):
        """Close session and connector"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Async GET request with retry logic
        
        Args:
            url: Target URL
            headers: Optional request headers
        
        Returns:
            Response data dict with 'status', 'data', 'headers', 'timestamp'
        """
        if not self.session:
            raise RuntimeError("Client not opened. Use 'async with' or call open()")
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(
                    url,
                    headers=headers or {},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    data = await response.text()
                    return {
                        'status': response.status,
                        'data': data,
                        'headers': dict(response.headers),
                        'timestamp': datetime.now(),
                        'attempt': attempt + 1
                    }
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Async POST request with retry logic
        
        Args:
            url: Target URL
            data: Request body
            headers: Optional request headers
        
        Returns:
            Response data dict
        """
        if not self.session:
            raise RuntimeError("Client not opened. Use 'async with' or call open()")
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    url,
                    json=data,
                    headers=headers or {},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    response_data = await response.text()
                    return {
                        'status': response.status,
                        'data': response_data,
                        'headers': dict(response.headers),
                        'timestamp': datetime.now(),
                        'attempt': attempt + 1
                    }
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))


class AsyncScraper:
    """
    Base class for async signal scrapers
    
    Subclasses should implement scrape_async() for non-blocking operations
    """
    
    def __init__(self, source_name: str, config: Optional[AsyncRequestConfig] = None):
        """Initialize async scraper"""
        self.source_name = source_name
        self.config = config or AsyncRequestConfig()
        self.logger = logging.getLogger(f"async.{source_name}")
    
    async def scrape_async(self) -> List[Dict[str, Any]]:
        """
        Async scrape implementation
        
        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement scrape_async()")


class AsyncBatchProcessor:
    """
    Process items concurrently with configurable concurrency limit
    
    Usage:
        processor = AsyncBatchProcessor(max_concurrent=5)
        results = await processor.process_batch(items, async_process_func)
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize batch processor
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(
        self,
        items: List[Any],
        async_func: Callable[[Any], Coroutine],
        on_error: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items concurrently
        
        Args:
            items: List of items to process
            async_func: Async function to apply to each item
            on_error: Optional error handler
        
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_task(item):
            async with semaphore:
                try:
                    return await async_func(item)
                except Exception as e:
                    if on_error:
                        await on_error(item, e)
                    self.logger.error(f"Error processing item: {e}")
                    return None
        
        tasks = [bounded_task(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return [r for r in results if r is not None]


class AsyncEventLoop:
    """
    Manages async event loop lifecycle
    
    Handles Python version differences and event loop policies
    """
    
    def __init__(self):
        """Initialize event loop manager"""
        self.logger = logging.getLogger(__name__)
    
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        
        return loop
    
    async def run_async(
        self,
        coro: Coroutine
    ) -> Any:
        """Run coroutine"""
        return await coro
    
    def run_until_complete(
        self,
        coro: Coroutine
    ) -> Any:
        """Run coroutine to completion"""
        loop = self.get_event_loop()
        return loop.run_until_complete(coro)


async def gather_with_limit(
    coros: List[Coroutine],
    max_concurrent: int = 5
) -> List[Any]:
    """
    Gather coroutines with concurrency limit
    
    Args:
        coros: List of coroutines
        max_concurrent: Maximum concurrent operations
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bounded_coro(c) for c in coros])


# Helper function to convert synchronous function to async
def async_wrapper(sync_func: Callable, *args, **kwargs) -> Coroutine:
    """
    Wrap synchronous function as async
    
    Useful for integrating sync code into async flows without blocking
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, sync_func, *args)
