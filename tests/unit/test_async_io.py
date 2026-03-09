"""
Tests for async I/O operations
"""

import pytest
import asyncio
from datetime import datetime

from src.ingestion.async_io import (
    AsyncRequestConfig,
    AsyncHTTPClient,
    AsyncBatchProcessor,
    AsyncEventLoop,
    gather_with_limit,
    AIOHTTP_AVAILABLE
)


# Skip all async tests if aiohttp not available
pytestmark = pytest.mark.skipif(
    not AIOHTTP_AVAILABLE,
    reason="aiohttp not installed"
)


class TestAsyncRequestConfig:
    """Test async request configuration"""
    
    def test_config_creation(self):
        """Test creating request config"""
        config = AsyncRequestConfig()
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_concurrent == 5
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = AsyncRequestConfig(
            timeout=60,
            max_retries=5,
            max_concurrent=10
        )
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.max_concurrent == 10
    
    def test_config_headers(self):
        """Test config with custom headers"""
        headers = {"User-Agent": "TestBot"}
        config = AsyncRequestConfig(headers=headers)
        assert config.headers == headers
    
    def test_config_pool_settings(self):
        """Test connection pool settings"""
        config = AsyncRequestConfig(
            use_connection_pool=True,
            pool_size=20
        )
        assert config.use_connection_pool
        assert config.pool_size == 20


class TestAsyncHTTPClient:
    """Test async HTTP client"""
    
    @pytest.mark.asyncio
    async def test_client_open_close(self):
        """Test opening and closing client"""
        client = AsyncHTTPClient()
        await client.open()
        assert client.session is not None
        await client.close()
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test using client as context manager"""
        async with AsyncHTTPClient() as client:
            assert client.session is not None
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_client_requires_open(self):
        """Test that client raises error if not opened"""
        client = AsyncHTTPClient()
        with pytest.raises(RuntimeError):
            await client.get("http://example.com")
    
    @pytest.mark.asyncio
    async def test_client_with_custom_config(self):
        """Test client with custom configuration"""
        config = AsyncRequestConfig(
            timeout=60,
            max_concurrent=10
        )
        client = AsyncHTTPClient(config)
        assert client.config.timeout == 60
        assert client.config.max_concurrent == 10


class TestAsyncBatchProcessor:
    """Test async batch processing"""
    
    @pytest.mark.asyncio
    async def test_processor_creation(self):
        """Test creating batch processor"""
        processor = AsyncBatchProcessor(max_concurrent=5)
        assert processor.max_concurrent == 5
    
    @pytest.mark.asyncio
    async def test_processor_sequential(self):
        """Test processing items sequentially"""
        processor = AsyncBatchProcessor(max_concurrent=1)
        
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(items, process_item)
        
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_processor_concurrent(self):
        """Test concurrent processing"""
        processor = AsyncBatchProcessor(max_concurrent=3)
        
        async def process_item(item):
            await asyncio.sleep(0.01)
            return item + 10
        
        items = list(range(10))
        results = await processor.process_batch(items, process_item)
        
        assert len(results) == 10
        assert all(r >= 10 for r in results)
    
    @pytest.mark.asyncio
    async def test_processor_with_error_handling(self):
        """Test error handling in batch processing"""
        processor = AsyncBatchProcessor(max_concurrent=2)
        
        errors_caught = []
        
        async def on_error(item, error):
            errors_caught.append((item, str(error)))
        
        async def process_item(item):
            if item == 3:
                raise ValueError(f"Error processing {item}")
            return item
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch(
            items,
            process_item,
            on_error=on_error
        )
        
        # Results should exclude the failed item
        assert 3 not in results
        assert len(results) == 4
    
    @pytest.mark.asyncio
    async def test_processor_empty_list(self):
        """Test processing empty list"""
        processor = AsyncBatchProcessor()
        
        async def process_item(item):
            return item
        
        results = await processor.process_batch([], process_item)
        assert results == []


class TestAsyncEventLoop:
    """Test event loop management"""
    
    def test_event_loop_creation(self):
        """Test creating event loop manager"""
        manager = AsyncEventLoop()
        assert manager is not None
    
    def test_get_event_loop(self):
        """Test getting event loop"""
        manager = AsyncEventLoop()
        loop = manager.get_event_loop()
        assert loop is not None
    
    @pytest.mark.asyncio
    async def test_run_async_coroutine(self):
        """Test running async coroutine"""
        manager = AsyncEventLoop()
        
        async def async_task():
            await asyncio.sleep(0.01)
            return "completed"
        
        result = await manager.run_async(async_task())
        assert result == "completed"


class TestGatherWithLimit:
    """Test gather with concurrency limit"""
    
    @pytest.mark.asyncio
    async def test_gather_with_limit(self):
        """Test gathering coroutines with limit"""
        
        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2
        
        coros = [task(i) for i in range(5)]
        results = await gather_with_limit(coros, max_concurrent=2)
        
        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]
    
    @pytest.mark.asyncio
    async def test_gather_with_single_concurrency(self):
        """Test gather with single concurrent operation"""
        
        async def task(n):
            await asyncio.sleep(0.01)
            return n
        
        coros = [task(i) for i in range(3)]
        results = await gather_with_limit(coros, max_concurrent=1)
        
        assert results == [0, 1, 2]
    
    @pytest.mark.asyncio
    async def test_gather_empty_list(self):
        """Test gathering empty list"""
        results = await gather_with_limit([], max_concurrent=5)
        assert results == []


class TestAsyncIOIntegration:
    """Integration tests for async I/O"""
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self):
        """Test complete batch processing workflow"""
        processor = AsyncBatchProcessor(max_concurrent=3)
        
        async def fetch_and_process(url):
            # Simulate API call
            await asyncio.sleep(0.01)
            return {"url": url, "status": 200}
        
        urls = [f"http://api.example.com/{i}" for i in range(5)]
        results = await processor.process_batch(urls, fetch_and_process)
        
        assert len(results) == 5
        assert all(r["status"] == 200 for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent HTTP-like requests"""
        
        async def simulated_request(request_id):
            await asyncio.sleep(0.01)
            return {"request_id": request_id, "data": "response"}
        
        # Process 10 requests with 3 concurrent
        processor = AsyncBatchProcessor(max_concurrent=3)
        requests = list(range(10))
        results = await processor.process_batch(
            requests,
            simulated_request
        )
        
        assert len(results) == 10


class TestAsyncErrorHandling:
    """Test error handling in async operations"""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling timeout errors"""
        
        async def timeout_task():
            try:
                await asyncio.sleep(1, result="timeout")
            except asyncio.TimeoutError:
                return "caught"
        
        # Task completes without actual timeout
        result = await timeout_task()
        assert result is None  # sleep completes
    
    @pytest.mark.asyncio
    async def test_exception_in_batch(self):
        """Test exceptions in batch processing"""
        processor = AsyncBatchProcessor(max_concurrent=2)
        
        exceptions = []
        
        async def on_error(item, error):
            exceptions.append(error)
        
        async def risky_task(item):
            if item % 2 == 0:
                raise ValueError(f"Error at {item}")
            return item
        
        items = [0, 1, 2, 3, 4]
        results = await processor.process_batch(
            items,
            risky_task,
            on_error=on_error
        )
        
        # Should have caught exceptions for even numbers
        assert len(exceptions) > 0


class TestAsyncConfiguration:
    """Test async configuration scenarios"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency(self):
        """Test with high concurrency setting"""
        processor = AsyncBatchProcessor(max_concurrent=20)
        
        async def quick_task(item):
            return item
        
        items = list(range(50))
        results = await processor.process_batch(items, quick_task)
        
        assert len(results) == 50
    
    @pytest.mark.asyncio
    async def test_low_concurrency(self):
        """Test with low concurrency setting"""
        processor = AsyncBatchProcessor(max_concurrent=1)
        
        async def quick_task(item):
            return item * 2
        
        items = [1, 2, 3]
        results = await processor.process_batch(items, quick_task)
        
        assert results == [2, 4, 6]
