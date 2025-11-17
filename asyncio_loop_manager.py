"""
Asyncio Event Loop Manager for Streamlit

This module provides a dedicated event loop running in a separate thread
to handle async MCP operations in Streamlit's synchronous execution model.
"""

import asyncio
import threading
import logging
import atexit
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


class AsyncioEventLoopThread:
    """
    Run event loop in a separate daemon thread.
    
    This allows async operations to be executed from Streamlit's synchronous
    context without blocking the main thread or conflicting with Streamlit's
    own event loop.
    """
    
    def __init__(self):
        """Initialize and start the event loop thread"""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        logger.info("AsyncioEventLoopThread: Started event loop in daemon thread")
    
    def _run_event_loop(self):
        """Run the event loop forever in the background thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        logger.info("AsyncioEventLoopThread: Event loop stopped")
    
    def run_coroutine(self, coroutine: Coroutine) -> Any:
        """
        Run a coroutine in the event loop and return the result.
        
        This method blocks until the coroutine completes, making it safe
        to use from synchronous Streamlit code.
        
        Args:
            coroutine: The async coroutine to execute
            
        Returns:
            The result of the coroutine execution
            
        Raises:
            Any exception raised by the coroutine
        """
        try:
            # Schedule the coroutine in the event loop thread
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            
            # Wait for the result (blocks current thread but not the event loop)
            result = future.result()
            
            return result
            
        except Exception as e:
            logger.error(f"AsyncioEventLoopThread: Error executing coroutine: {e}", exc_info=True)
            raise
    
    def close(self):
        """Close the event loop and stop the thread"""
        try:
            if self.loop:
                if self.loop.is_running():
                    logger.info("AsyncioEventLoopThread: Stopping event loop")
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    
                    # Wait for thread to finish with timeout
                    self.thread.join(timeout=5)
                    
                    if self.thread.is_alive():
                        logger.warning("AsyncioEventLoopThread: Thread did not stop within timeout")
                
                # Close the loop if not already closed
                if not self.loop.is_closed():
                    self.loop.close()
                    logger.info("AsyncioEventLoopThread: Event loop closed")
                    
        except Exception as e:
            logger.error(f"AsyncioEventLoopThread: Error closing event loop: {e}", exc_info=True)


def get_or_create_event_loop_thread():
    """
    Get or create the event loop thread from Streamlit session state.
    
    This ensures a single event loop thread is shared across Streamlit reruns.
    
    Returns:
        AsyncioEventLoopThread instance
    """
    import streamlit as st
    
    if 'asyncio_loop_thread' not in st.session_state:
        logger.info("Creating new AsyncioEventLoopThread in session state")
        st.session_state.asyncio_loop_thread = AsyncioEventLoopThread()
        
        # Register cleanup function for program exit
        def cleanup_resources():
            if 'asyncio_loop_thread' in st.session_state:
                logger.info("Cleaning up AsyncioEventLoopThread on exit")
                st.session_state.asyncio_loop_thread.close()
        
        atexit.register(cleanup_resources)
    
    return st.session_state.asyncio_loop_thread
