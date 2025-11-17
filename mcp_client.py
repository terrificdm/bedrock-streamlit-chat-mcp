"""
MCP Client Connection Management Module

This module handles connections to MCP servers, tool discovery, and tool execution.
It provides classes for managing individual server connections and orchestrating
multiple MCP servers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from mcp_config import MCPServerConfig, MCPConfig, load_config

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConnection:
    """Represents a connection to a single MCP server"""
    
    config: MCPServerConfig
    client: Optional[Any] = None
    session: Optional[ClientSession] = None
    tools: List[Dict] = field(default_factory=list)
    connected: bool = False
    error_message: Optional[str] = None
    
    async def connect(self) -> bool:
        """
        Establish connection to MCP server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"MCP: Attempting to connect to server '{self.config.name}'")
            logger.info(f"MCP: Connection type: {self.config.connection_type}")
            
            # Connect based on connection type
            if self.config.connection_type == "stdio":
                return await self._connect_stdio()
            elif self.config.connection_type == "sse":
                return await self._connect_sse()
            elif self.config.connection_type == "http":
                return await self._connect_http()
            else:
                self.connected = False
                self.error_message = f"Unsupported connection type: {self.config.connection_type}"
                logger.error(f"MCP: Server '{self.config.name}' - {self.error_message}")
                return False
                
        except Exception as e:
            self.connected = False
            self.error_message = f"Unexpected error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Unexpected connection error: {e}", exc_info=True)
            logger.error(f"MCP: Please check the server configuration and logs")
            return False
    
    async def _connect_stdio(self) -> bool:
        """Establish stdio connection to MCP server"""
        try:
            logger.info(f"MCP: Server '{self.config.name}' - Using stdio connection")
            logger.info(f"MCP: Command: {self.config.command} {' '.join(self.config.args)}")
            
            # Prepare environment: for uv-based servers, we need to handle VIRTUAL_ENV carefully
            server_env = None
            
            if self.config.env:
                # Use configured environment
                server_env = dict(self.config.env)
            
            # For uv-based servers, inherit parent env but remove VIRTUAL_ENV
            # to avoid conflicts with the server's own virtual environment
            if self.config.command in ["uv", "uvx"]:
                import os
                server_env = dict(os.environ)
                # Remove VIRTUAL_ENV to let uv manage its own environment
                server_env.pop('VIRTUAL_ENV', None)
                # Merge in any configured env vars
                if self.config.env:
                    server_env.update(self.config.env)
            
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=server_env
            )
            
            # Establish stdio connection with timeout
            try:
                # stdio_client returns an async context manager
                # We need to enter it and keep it alive for the connection lifetime
                self.client = stdio_client(server_params)
                read_stream, write_stream = await asyncio.wait_for(
                    self.client.__aenter__(), 
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - stdio connection established")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Connection timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Connection timeout after 10 seconds")
                return False
            except FileNotFoundError as e:
                self.connected = False
                self.error_message = f"Command not found: {self.config.command}"
                logger.error(f"MCP: Server '{self.config.name}' - Command '{self.config.command}' not found")
                logger.error(f"MCP: Please ensure the command is installed and in your PATH")
                return False
            except PermissionError as e:
                self.connected = False
                self.error_message = f"Permission denied: {self.config.command}"
                logger.error(f"MCP: Server '{self.config.name}' - Permission denied for command '{self.config.command}'")
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Connection failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - Connection failed: {e}", exc_info=True)
                return False
            
            # Create and initialize client session with timeout
            # ClientSession must be used as an async context manager
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Creating and initializing client session")
                session_cm = ClientSession(read_stream, write_stream)
                self.session = await asyncio.wait_for(
                    session_cm.__aenter__(),
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - Session initialized successfully")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Session initialization timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Session initialization timeout after 10 seconds")
                logger.error(f"MCP: The server process started but did not complete the MCP handshake")
                # Clean up the stdio client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Session initialization failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - Failed to initialize session: {e}", exc_info=True)
                logger.error(f"MCP: The server may have crashed or returned invalid data")
                # Clean up the stdio client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            
            # List available tools (with built-in retry logic)
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Listing available tools")
                await self._refresh_tools()
            except Exception as e:
                # Connection succeeded but tool listing failed - still mark as connected
                logger.warning(f"MCP: Server '{self.config.name}' - Failed to list tools: {e}")
                logger.warning(f"MCP: Server is connected but may not be functioning correctly")
            
            self.connected = True
            self.error_message = None
            logger.info(f"MCP: Successfully connected to '{self.config.name}' with {len(self.tools)} tool(s)")
            return True
            
        except asyncio.TimeoutError as e:
            self.connected = False
            self.error_message = "Connection timeout"
            logger.error(f"MCP: Server '{self.config.name}' - Connection timeout")
            logger.error(f"MCP: The server may be unresponsive or taking too long to start")
            return False
        except ConnectionError as e:
            self.connected = False
            self.error_message = f"Connection error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Connection error: {e}")
            return False
        except Exception as e:
            self.connected = False
            self.error_message = f"Unexpected error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Unexpected connection error: {e}", exc_info=True)
            logger.error(f"MCP: Please check the server configuration and logs")
            return False
    
    async def _connect_sse(self) -> bool:
        """Establish SSE/HTTP connection to MCP server"""
        try:
            logger.info(f"MCP: Server '{self.config.name}' - Using SSE/HTTP connection")
            logger.info(f"MCP: URL: {self.config.url}")
            
            # Establish SSE connection with timeout
            try:
                # sse_client returns an async context manager
                self.client = sse_client(self.config.url)
                read_stream, write_stream = await asyncio.wait_for(
                    self.client.__aenter__(),
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - SSE connection established")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Connection timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Connection timeout after 10 seconds")
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Connection failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - SSE connection failed: {e}", exc_info=True)
                return False
            
            # Create and initialize client session with timeout
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Creating and initializing client session")
                session_cm = ClientSession(read_stream, write_stream)
                self.session = await asyncio.wait_for(
                    session_cm.__aenter__(),
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - Session initialized successfully")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Session initialization timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Session initialization timeout after 10 seconds")
                logger.error(f"MCP: The server responded but did not complete the MCP handshake")
                # Clean up the SSE client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Session initialization failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - Failed to initialize session: {e}", exc_info=True)
                logger.error(f"MCP: The server may have returned invalid data")
                # Clean up the SSE client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            
            # List available tools
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Listing available tools")
                await self._refresh_tools()
            except Exception as e:
                # Connection succeeded but tool listing failed - still mark as connected
                logger.warning(f"MCP: Server '{self.config.name}' - Failed to list tools: {e}")
                logger.warning(f"MCP: Server is connected but may not be functioning correctly")
            
            self.connected = True
            self.error_message = None
            logger.info(f"MCP: Successfully connected to '{self.config.name}' with {len(self.tools)} tool(s)")
            return True
            
        except asyncio.TimeoutError as e:
            self.connected = False
            self.error_message = "Connection timeout"
            logger.error(f"MCP: Server '{self.config.name}' - Connection timeout")
            logger.error(f"MCP: The server may be unresponsive or taking too long to respond")
            return False
        except ConnectionError as e:
            self.connected = False
            self.error_message = f"Connection error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Connection error: {e}")
            return False
        except Exception as e:
            self.connected = False
            self.error_message = f"Unexpected error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Unexpected SSE connection error: {e}", exc_info=True)
            logger.error(f"MCP: Please check the server URL and configuration")
            return False
    
    async def _connect_http(self) -> bool:
        """Establish StreamableHTTP connection to MCP server"""
        try:
            logger.info(f"MCP: Server '{self.config.name}' - Using StreamableHTTP connection")
            logger.info(f"MCP: URL: {self.config.url}")
            
            # Establish StreamableHTTP connection with timeout
            try:
                # streamablehttp_client returns an async context manager
                self.client = streamablehttp_client(self.config.url)
                async_gen = self.client.__aenter__()
                read_stream, write_stream, get_session_id = await asyncio.wait_for(
                    async_gen,
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - StreamableHTTP connection established")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Connection timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Connection timeout after 10 seconds")
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Connection failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - StreamableHTTP connection failed: {e}", exc_info=True)
                return False
            
            # Create and initialize client session with timeout
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Creating and initializing client session")
                session_cm = ClientSession(read_stream, write_stream)
                self.session = await asyncio.wait_for(
                    session_cm.__aenter__(),
                    timeout=10.0
                )
                logger.info(f"MCP: Server '{self.config.name}' - Session initialized successfully")
            except asyncio.TimeoutError:
                self.connected = False
                self.error_message = "Session initialization timeout (10s)"
                logger.error(f"MCP: Server '{self.config.name}' - Session initialization timeout after 10 seconds")
                logger.error(f"MCP: The server responded but did not complete the MCP handshake")
                # Clean up the HTTP client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            except Exception as e:
                self.connected = False
                self.error_message = f"Session initialization failed: {str(e)}"
                logger.error(f"MCP: Server '{self.config.name}' - Failed to initialize session: {e}", exc_info=True)
                logger.error(f"MCP: The server may have returned invalid data")
                # Clean up the HTTP client
                if self.client:
                    try:
                        await self.client.__aexit__(None, None, None)
                    except:
                        pass
                return False
            
            # List available tools
            try:
                logger.info(f"MCP: Server '{self.config.name}' - Listing available tools")
                await self._refresh_tools()
            except Exception as e:
                # Connection succeeded but tool listing failed - still mark as connected
                logger.warning(f"MCP: Server '{self.config.name}' - Failed to list tools: {e}")
                logger.warning(f"MCP: Server is connected but may not be functioning correctly")
            
            self.connected = True
            self.error_message = None
            logger.info(f"MCP: Successfully connected to '{self.config.name}' with {len(self.tools)} tool(s)")
            return True
            
        except asyncio.TimeoutError as e:
            self.connected = False
            self.error_message = "Connection timeout"
            logger.error(f"MCP: Server '{self.config.name}' - Connection timeout")
            logger.error(f"MCP: The server may be unresponsive or taking too long to respond")
            return False
        except ConnectionError as e:
            self.connected = False
            self.error_message = f"Connection error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Connection error: {e}")
            return False
        except Exception as e:
            self.connected = False
            self.error_message = f"Unexpected error: {str(e)}"
            logger.error(f"MCP: Server '{self.config.name}' - Unexpected HTTP connection error: {e}", exc_info=True)
            logger.error(f"MCP: Please check the server URL and configuration")
            return False
    
    async def disconnect(self):
        """Close connection to MCP server"""
        try:
            logger.info(f"MCP: Disconnecting from server '{self.config.name}'")
            
            # Close the session (context manager)
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except RuntimeError as e:
                    # Ignore "cancel scope in different task" errors - this is expected
                    # when disconnecting from a different asyncio task than connection
                    if "cancel scope" not in str(e).lower():
                        logger.warning(f"MCP: Error closing session for '{self.config.name}': {e}")
                except Exception as e:
                    logger.warning(f"MCP: Error closing session for '{self.config.name}': {e}")
                finally:
                    self.session = None
            
            # Close the stdio client (context manager)
            if self.client:
                try:
                    await self.client.__aexit__(None, None, None)
                except RuntimeError as e:
                    # Ignore "cancel scope in different task" errors - this is expected
                    # when disconnecting from a different asyncio task than connection
                    if "cancel scope" not in str(e).lower():
                        logger.warning(f"MCP: Error closing client for '{self.config.name}': {e}")
                except Exception as e:
                    logger.warning(f"MCP: Error closing client for '{self.config.name}': {e}")
                finally:
                    self.client = None
            
            self.connected = False
            self.tools = []
            logger.info(f"MCP: Successfully disconnected from server '{self.config.name}'")
            
        except Exception as e:
            logger.error(f"MCP: Unexpected error disconnecting from '{self.config.name}': {e}", exc_info=True)
            # Still mark as disconnected even if cleanup failed
            self.connected = False
            self.session = None
            self.client = None
    
    async def _refresh_tools(self):
        """Retrieve and cache available tools from server with retry logic"""
        if not self.session:
            return
        
        # Retry logic for servers that need initialization time
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # List tools from the server
                tools_response = await self.session.list_tools(cursor=None)
                
                # Convert tools to dictionary format
                self.tools = []
                if tools_response and hasattr(tools_response, 'tools'):
                    for tool in tools_response.tools:
                        tool_dict = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                        self.tools.append(tool_dict)
                
                logger.info(f"MCP: Retrieved {len(self.tools)} tools from '{self.config.name}'")
                return  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's an initialization timing issue or invalid request parameters
                # (which often means the server isn't ready yet)
                is_timing_issue = ("initialization" in error_msg or 
                                 "invalid request parameters" in error_msg or
                                 "not ready" in error_msg)
                
                if is_timing_issue and attempt < max_retries - 1:
                    logger.info(f"MCP: Server '{self.config.name}' not ready yet, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue  # Try again
                else:
                    logger.error(f"MCP: Failed to list tools from '{self.config.name}': {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"MCP: Gave up after {max_retries} attempts")
                    self.tools = []
                    return
    
    async def list_tools(self) -> List[Dict]:
        """
        Get available tools from server
        
        Returns:
            List of tool definitions
        """
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Execute a tool on the server
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool input parameters
            
        Returns:
            Tool execution result
            
        Raises:
            Exception if tool execution fails
        """
        if not self.session or not self.connected:
            error_msg = f"Not connected to server '{self.config.name}'"
            logger.error(f"MCP: {error_msg}")
            raise ConnectionError(error_msg)
        
        try:
            logger.info(f"MCP: Executing tool '{tool_name}' on server '{self.config.name}'")
            logger.debug(f"MCP: Tool arguments: {arguments}")
            
            # Call the tool with timeout
            try:
                result = await asyncio.wait_for(
                    self.session.call_tool(tool_name, arguments),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                error_msg = f"Tool '{tool_name}' execution timed out after 30 seconds"
                logger.error(f"MCP: {error_msg}")
                raise TimeoutError(error_msg)
            
            logger.info(f"MCP: Tool '{tool_name}' executed successfully")
            logger.debug(f"MCP: Tool result: {result}")
            return result
            
        except TimeoutError:
            # Re-raise timeout errors
            raise
        except ConnectionError:
            # Re-raise connection errors
            raise
        except ValueError as e:
            # Handle invalid arguments
            error_msg = f"Invalid arguments for tool '{tool_name}': {str(e)}"
            logger.error(f"MCP: {error_msg}")
            raise ValueError(error_msg)
        except Exception as e:
            # Handle all other errors
            error_msg = f"Tool '{tool_name}' execution failed: {str(e)}"
            logger.error(f"MCP: {error_msg}", exc_info=True)
            raise RuntimeError(error_msg)


class MCPClientManager:
    """Manages all MCP server connections and tool routing"""
    
    def __init__(self, config_path: str = "mcp.json"):
        """
        Initialize MCP Client Manager
        
        Args:
            config_path: Path to MCP configuration file
        """
        self.config_path = config_path
        self.connections: Dict[str, MCPServerConnection] = {}
        self.tool_registry: Dict[str, MCPServerConnection] = {}
        self.config: Optional[MCPConfig] = None
    
    async def initialize(self):
        """Load configuration and connect to all enabled servers"""
        try:
            logger.info(f"MCP: Starting initialization from {self.config_path}")
            
            # Load configuration
            self.config = load_config(self.config_path)
            
            if self.config is None:
                logger.warning("MCP: No valid configuration found, MCP features disabled")
                return
            
            # Get enabled servers
            enabled_servers = self.config.get_enabled_servers()
            
            if not enabled_servers:
                logger.info("MCP: No enabled servers in configuration")
                logger.info("MCP: Edit mcp.json and set 'disabled: false' to enable servers")
                return
            
            logger.info(f"MCP: Attempting to connect to {len(enabled_servers)} enabled server(s)")
            
            # Track connection results
            successful_connections = 0
            failed_connections = 0
            
            # Connect to each enabled server
            for server_config in enabled_servers:
                try:
                    logger.info(f"MCP: Processing server '{server_config.name}'")
                    connection = MCPServerConnection(config=server_config)
                    
                    # Attempt connection
                    success = await connection.connect()
                    
                    # Store connection regardless of success (for status display)
                    self.connections[server_config.name] = connection
                    
                    if success:
                        successful_connections += 1
                        
                        # Build tool registry for successful connections
                        for tool in connection.tools:
                            tool_name = tool["name"]
                            
                            # Check for tool name conflicts
                            if tool_name in self.tool_registry:
                                logger.warning(
                                    f"MCP: Tool '{tool_name}' from server '{server_config.name}' "
                                    f"conflicts with existing tool from '{self.tool_registry[tool_name].config.name}'. "
                                    f"Using tool from '{self.tool_registry[tool_name].config.name}'"
                                )
                            else:
                                self.tool_registry[tool_name] = connection
                                logger.debug(f"MCP: Registered tool '{tool_name}' from server '{server_config.name}'")
                    else:
                        failed_connections += 1
                        logger.warning(f"MCP: Server '{server_config.name}' failed to connect, continuing with remaining servers")
                        
                except Exception as e:
                    failed_connections += 1
                    logger.error(f"MCP: Unexpected error processing server '{server_config.name}': {e}", exc_info=True)
                    # Create a disconnected connection for status display
                    connection = MCPServerConnection(config=server_config)
                    connection.connected = False
                    connection.error_message = f"Initialization error: {str(e)}"
                    self.connections[server_config.name] = connection
            
            # Log summary
            total_servers = len(enabled_servers)
            logger.info(f"MCP: Initialization complete")
            logger.info(f"MCP: Total servers: {total_servers}, Connected: {successful_connections}, Failed: {failed_connections}")
            logger.info(f"MCP: Total tools available: {len(self.tool_registry)}")
            
            if successful_connections == 0 and total_servers > 0:
                logger.error("MCP: No servers connected successfully. MCP features will not be available.")
            elif failed_connections > 0:
                logger.warning(f"MCP: {failed_connections} server(s) failed to connect. Some tools may be unavailable.")
            
        except Exception as e:
            logger.error(f"MCP: Critical error during initialization: {e}", exc_info=True)
            logger.error(f"MCP: MCP features will not be available")
    
    async def shutdown(self):
        """Close all server connections"""
        logger.info("MCP: Shutting down all connections")
        
        for connection in self.connections.values():
            if connection.connected:
                await connection.disconnect()
        
        self.connections.clear()
        self.tool_registry.clear()
        logger.info("MCP: Shutdown complete")
    
    async def disconnect_all(self):
        """Disconnect all connected servers without clearing connections"""
        logger.info("MCP: Disconnecting all servers")
        
        disconnected_count = 0
        for connection in self.connections.values():
            if connection.connected:
                try:
                    await connection.disconnect()
                    disconnected_count += 1
                except Exception as e:
                    logger.error(f"MCP: Error disconnecting server '{connection.config.name}': {e}")
        
        # Clear tool registry
        self.tool_registry.clear()
        
        logger.info(f"MCP: Disconnected {disconnected_count} server(s)")
        return disconnected_count
    
    async def reconnect_all(self):
        """Reconnect all enabled servers"""
        logger.info("MCP: Reconnecting all enabled servers")
        
        if not self.config:
            logger.error("MCP: No configuration available for reconnection")
            return 0, 0
        
        # Get enabled servers from config
        enabled_servers = self.config.get_enabled_servers()
        
        if not enabled_servers:
            logger.info("MCP: No enabled servers to reconnect")
            return 0, 0
        
        logger.info(f"MCP: Attempting to reconnect {len(enabled_servers)} enabled server(s)")
        
        # First, disconnect all current connections
        await self.disconnect_all()
        
        # Track connection results
        successful_connections = 0
        failed_connections = 0
        
        # Reconnect to each enabled server
        for server_config in enabled_servers:
            try:
                logger.info(f"MCP: Reconnecting to server '{server_config.name}'")
                
                # Create new connection
                connection = MCPServerConnection(config=server_config)
                
                # Attempt connection
                success = await connection.connect()
                
                # Store connection
                self.connections[server_config.name] = connection
                
                if success:
                    successful_connections += 1
                    
                    # Rebuild tool registry
                    for tool in connection.tools:
                        tool_name = tool["name"]
                        
                        # Check for tool name conflicts
                        if tool_name in self.tool_registry:
                            logger.warning(
                                f"MCP: Tool '{tool_name}' from server '{server_config.name}' "
                                f"conflicts with existing tool from '{self.tool_registry[tool_name].config.name}'. "
                                f"Using tool from '{self.tool_registry[tool_name].config.name}'"
                            )
                        else:
                            self.tool_registry[tool_name] = connection
                else:
                    failed_connections += 1
                    logger.warning(f"MCP: Server '{server_config.name}' failed to reconnect")
                    
            except Exception as e:
                failed_connections += 1
                logger.error(f"MCP: Unexpected error reconnecting server '{server_config.name}': {e}", exc_info=True)
                # Create a disconnected connection for status display
                connection = MCPServerConnection(config=server_config)
                connection.connected = False
                connection.error_message = f"Reconnection error: {str(e)}"
                self.connections[server_config.name] = connection
        
        # Log summary
        logger.info(f"MCP: Reconnection complete")
        logger.info(f"MCP: Connected: {successful_connections}, Failed: {failed_connections}")
        logger.info(f"MCP: Total tools available: {len(self.tool_registry)}")
        
        return successful_connections, failed_connections
    
    def get_all_tools(self) -> List[Dict]:
        """
        Get all tools from all connected servers
        
        Returns:
            List of tool definitions from all connected servers
        """
        all_tools = []
        
        for connection in self.connections.values():
            if connection.connected:
                all_tools.extend(connection.tools)
        
        return all_tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Route and execute tool call to appropriate server
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool input parameters
            
        Returns:
            Dictionary containing tool result with status
        """
        try:
            logger.info(f"MCP: Routing tool call for '{tool_name}'")
            
            # Find the server that provides this tool
            if tool_name not in self.tool_registry:
                error_msg = f"Tool '{tool_name}' not found in any connected server"
                logger.error(f"MCP: {error_msg}")
                logger.error(f"MCP: Available tools: {list(self.tool_registry.keys())}")
                return {
                    "status": "error",
                    "content": error_msg
                }
            
            connection = self.tool_registry[tool_name]
            
            # Check if connection is still active
            if not connection.connected:
                error_msg = f"Server '{connection.config.name}' is not connected"
                logger.error(f"MCP: {error_msg}")
                return {
                    "status": "error",
                    "content": error_msg
                }
            
            logger.info(f"MCP: Executing tool '{tool_name}' on server '{connection.config.name}'")
            
            # Execute the tool
            try:
                result = await connection.call_tool(tool_name, arguments)
                
                logger.info(f"MCP: Tool '{tool_name}' completed successfully")
                return {
                    "status": "success",
                    "content": result
                }
                
            except TimeoutError as e:
                error_msg = f"Tool '{tool_name}' timed out: {str(e)}"
                logger.error(f"MCP: {error_msg}")
                return {
                    "status": "error",
                    "content": error_msg
                }
            except ConnectionError as e:
                error_msg = f"Connection error executing '{tool_name}': {str(e)}"
                logger.error(f"MCP: {error_msg}")
                # Mark connection as disconnected
                connection.connected = False
                connection.error_message = str(e)
                return {
                    "status": "error",
                    "content": error_msg
                }
            except ValueError as e:
                error_msg = f"Invalid arguments for '{tool_name}': {str(e)}"
                logger.error(f"MCP: {error_msg}")
                return {
                    "status": "error",
                    "content": error_msg
                }
            
        except Exception as e:
            error_msg = f"Unexpected error executing tool '{tool_name}': {str(e)}"
            logger.error(f"MCP: {error_msg}", exc_info=True)
            return {
                "status": "error",
                "content": error_msg
            }
    
    def get_server_status(self) -> List[Dict]:
        """
        Get connection status of all servers for UI display
        
        Returns:
            List of server status dictionaries
        """
        status_list = []
        
        for server_name, connection in self.connections.items():
            status = {
                "name": server_name,
                "connected": connection.connected,
                "tools": connection.tools if connection.connected else [],
                "error": connection.error_message
            }
            status_list.append(status)
        
        return status_list
    
    def convert_tools_to_bedrock_format(self) -> Optional[Dict]:
        """
        Convert MCP tool schemas to Bedrock toolConfig format
        
        Returns:
            Dictionary in Bedrock toolConfig format, or None if no tools available
        """
        try:
            all_tools = self.get_all_tools()
            
            if not all_tools:
                logger.info("MCP: No tools available to convert to Bedrock format")
                return None
            
            logger.info(f"MCP: Converting {len(all_tools)} tool(s) to Bedrock format")
            tool_specs = []
            skipped_tools = 0
            
            for tool in all_tools:
                try:
                    # Validate required fields
                    if "name" not in tool:
                        logger.warning(f"MCP: Tool missing 'name' field, skipping")
                        skipped_tools += 1
                        continue
                    
                    tool_name = tool["name"]
                    
                    # Validate tool name format (Bedrock requirements)
                    if not tool_name or not isinstance(tool_name, str):
                        logger.warning(f"MCP: Tool has invalid name format, skipping")
                        skipped_tools += 1
                        continue
                    
                    # Get and validate description
                    description = tool.get("description", "")
                    if not isinstance(description, str):
                        logger.warning(f"MCP: Tool '{tool_name}' has invalid description, using empty string")
                        description = ""
                    
                    # Get and validate input schema
                    input_schema = tool.get("inputSchema", {})
                    if not isinstance(input_schema, dict):
                        logger.warning(
                            f"MCP: Tool '{tool_name}' has invalid inputSchema type, using empty schema"
                        )
                        input_schema = {
                            "type": "object",
                            "properties": {}
                        }
                    
                    # Ensure schema has required fields for Bedrock
                    if "type" not in input_schema:
                        logger.warning(f"MCP: Tool '{tool_name}' inputSchema missing 'type', adding 'object'")
                        input_schema["type"] = "object"
                    
                    if "properties" not in input_schema and input_schema.get("type") == "object":
                        logger.debug(f"MCP: Tool '{tool_name}' inputSchema missing 'properties', adding empty dict")
                        input_schema["properties"] = {}
                    
                    # Build Bedrock toolSpec
                    tool_spec = {
                        "toolSpec": {
                            "name": tool_name,
                            "description": description,
                            "inputSchema": {
                                "json": input_schema
                            }
                        }
                    }
                    
                    tool_specs.append(tool_spec)
                    logger.debug(f"MCP: Successfully converted tool '{tool_name}' to Bedrock format")
                    
                except KeyError as e:
                    logger.error(f"MCP: Missing required field in tool definition: {e}")
                    skipped_tools += 1
                    continue
                except Exception as e:
                    logger.error(f"MCP: Error converting tool to Bedrock format: {e}", exc_info=True)
                    skipped_tools += 1
                    continue
            
            if not tool_specs:
                logger.warning("MCP: No valid tools to convert to Bedrock format")
                if skipped_tools > 0:
                    logger.warning(f"MCP: Skipped {skipped_tools} invalid tool(s)")
                return None
            
            tool_config = {
                "tools": tool_specs
            }
            
            logger.info(f"MCP: Successfully converted {len(tool_specs)} tool(s) to Bedrock format")
            if skipped_tools > 0:
                logger.warning(f"MCP: Skipped {skipped_tools} invalid tool(s)")
            
            return tool_config
            
        except Exception as e:
            logger.error(f"MCP: Critical error converting tools to Bedrock format: {e}", exc_info=True)
            return None
