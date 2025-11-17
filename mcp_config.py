"""
MCP Configuration Management Module

This module handles loading, validating, and managing MCP server configurations
from the mcp.json configuration file.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def expand_env_vars(value: str) -> str:
    """
    Expand environment variables in a string.
    Supports ${VAR_NAME} syntax.
    
    Args:
        value: String potentially containing environment variable references
        
    Returns:
        String with environment variables expanded
    """
    def replace_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            logger.warning(f"MCP: Environment variable '{var_name}' not found, using empty string")
            return ""
        return env_value
    
    # Replace ${VAR_NAME} with environment variable value
    return re.sub(r'\$\{([^}]+)\}', replace_var, value)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    disabled: bool = False
    connection_type: Optional[str] = None  # stdio, sse, http - auto-detected if not specified
    
    @classmethod
    def from_dict(cls, name: str, config_dict: dict) -> 'MCPServerConfig':
        """Create MCPServerConfig from dictionary with environment variable expansion"""
        # Expand environment variables in env dict
        env = config_dict.get("env", {})
        expanded_env = {key: expand_env_vars(value) for key, value in env.items()}
        
        # Auto-detect connection type if not specified
        connection_type = config_dict.get("connection_type")
        if connection_type is None:
            if "url" in config_dict:
                connection_type = "http"  # Default to StreamableHTTP for URL-based connections (recommended)
            else:
                connection_type = "stdio"  # Default to stdio for command-based connections
        
        return cls(
            name=name,
            command=config_dict.get("command"),
            args=config_dict.get("args"),
            url=config_dict.get("url"),
            env=expanded_env,
            disabled=config_dict.get("disabled", False),
            connection_type=connection_type
        )


@dataclass
class MCPConfig:
    """Complete MCP configuration containing all server configs"""
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled servers"""
        return [server for server in self.servers.values() if not server.disabled]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MCPConfig':
        """Create MCPConfig from dictionary"""
        servers = {}
        mcp_servers = config_dict.get("mcpServers", {})
        
        for name, server_config in mcp_servers.items():
            servers[name] = MCPServerConfig.from_dict(name, server_config)
        
        return cls(servers=servers)


def validate_config(config_dict: dict) -> bool:
    """
    Validate MCP configuration structure
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        if not isinstance(config_dict, dict):
            logger.error("MCP: Configuration must be a dictionary/object")
            logger.error("MCP: Expected format: {\"mcpServers\": {...}}")
            return False
        
        if "mcpServers" not in config_dict:
            logger.error("MCP: Configuration missing required 'mcpServers' key")
            logger.error("MCP: Expected format: {\"mcpServers\": {\"server_name\": {...}}}")
            return False
        
        mcp_servers = config_dict["mcpServers"]
        if not isinstance(mcp_servers, dict):
            logger.error("MCP: 'mcpServers' must be a dictionary/object")
            logger.error("MCP: Expected format: {\"mcpServers\": {\"server_name\": {...}}}")
            return False
        
        if len(mcp_servers) == 0:
            logger.warning("MCP: No servers defined in configuration")
            return True  # Empty config is valid, just not useful
        
        # Validate each server configuration
        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                logger.error(f"MCP: Server config for '{server_name}' must be a dictionary/object")
                return False
            
            # Check that server has either command+args OR url
            has_command = "command" in server_config and "args" in server_config
            has_url = "url" in server_config
            
            if not has_command and not has_url:
                logger.error(f"MCP: Server '{server_name}' must have either 'command'+'args' (stdio) or 'url' (HTTP/SSE)")
                logger.error(f"MCP: Stdio example: \"command\": \"npx\", \"args\": [\"...\"]")
                logger.error(f"MCP: HTTP/SSE example: \"url\": \"https://example.com\"")
                return False
            
            if has_command and has_url:
                logger.error(f"MCP: Server '{server_name}' cannot have both 'command' and 'url'")
                logger.error(f"MCP: Use 'command'+'args' for stdio OR 'url' for HTTP/SSE, not both")
                return False
            
            # Validate stdio configuration
            if has_command:
                if not isinstance(server_config["command"], str):
                    logger.error(f"MCP: Server '{server_name}' command must be a string")
                    logger.error(f"MCP: Got: {type(server_config['command']).__name__}")
                    return False
                
                if server_config["command"].strip() == "":
                    logger.error(f"MCP: Server '{server_name}' command cannot be empty")
                    return False
                
                if not isinstance(server_config["args"], list):
                    logger.error(f"MCP: Server '{server_name}' args must be a list/array")
                    logger.error(f"MCP: Got: {type(server_config['args']).__name__}")
                    return False
                
                # Validate args are strings
                for i, arg in enumerate(server_config["args"]):
                    if not isinstance(arg, str):
                        logger.error(f"MCP: Server '{server_name}' args[{i}] must be a string")
                        logger.error(f"MCP: Got: {type(arg).__name__}")
                        return False
            
            # Validate HTTP/SSE configuration
            if has_url:
                if not isinstance(server_config["url"], str):
                    logger.error(f"MCP: Server '{server_name}' url must be a string")
                    logger.error(f"MCP: Got: {type(server_config['url']).__name__}")
                    return False
                
                if server_config["url"].strip() == "":
                    logger.error(f"MCP: Server '{server_name}' url cannot be empty")
                    return False
                
                # Basic URL validation
                url = server_config["url"].strip()
                if not (url.startswith("http://") or url.startswith("https://")):
                    logger.error(f"MCP: Server '{server_name}' url must start with http:// or https://")
                    logger.error(f"MCP: Got: {url}")
                    return False
            
            # Validate optional fields if present
            if "env" in server_config:
                if not isinstance(server_config["env"], dict):
                    logger.error(f"MCP: Server '{server_name}' env must be a dictionary/object")
                    logger.error(f"MCP: Got: {type(server_config['env']).__name__}")
                    return False
                
                # Validate env values are strings
                for key, value in server_config["env"].items():
                    if not isinstance(value, str):
                        logger.error(f"MCP: Server '{server_name}' env['{key}'] must be a string")
                        logger.error(f"MCP: Got: {type(value).__name__}")
                        return False
            
            if "disabled" in server_config and not isinstance(server_config["disabled"], bool):
                logger.error(f"MCP: Server '{server_name}' disabled must be a boolean (true/false)")
                logger.error(f"MCP: Got: {type(server_config['disabled']).__name__}")
                return False
            
            if "connection_type" in server_config:
                if not isinstance(server_config["connection_type"], str):
                    logger.error(f"MCP: Server '{server_name}' connection_type must be a string")
                    return False
                
                valid_types = ["stdio", "sse", "http"]
                if server_config["connection_type"] not in valid_types:
                    logger.error(f"MCP: Server '{server_name}' connection_type must be one of: {valid_types}")
                    logger.error(f"MCP: Got: {server_config['connection_type']}")
                    return False
        
        logger.info(f"MCP: Configuration validation passed for {len(mcp_servers)} server(s)")
        return True
        
    except Exception as e:
        logger.error(f"MCP: Unexpected error during configuration validation: {e}", exc_info=True)
        return False


def load_config(config_path: str = "mcp.json") -> Optional[MCPConfig]:
    """
    Load and parse MCP configuration from file
    
    Args:
        config_path: Path to mcp.json configuration file
        
    Returns:
        MCPConfig object if successful, None if file doesn't exist or is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"MCP: Configuration file not found at {config_path}")
        logger.info(f"MCP: Creating default configuration file at {config_path}")
        
        # Create default config if missing
        if save_default_config(config_path):
            logger.info(f"MCP: Default configuration created. Please edit {config_path} to configure your servers.")
        else:
            logger.error(f"MCP: Failed to create default configuration file")
        
        return None
    
    try:
        logger.info(f"MCP: Loading configuration from {config_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        if not validate_config(config_dict):
            logger.error(f"MCP: Invalid configuration structure in {config_path}")
            logger.error(f"MCP: Please check the configuration file format and required fields")
            return None
        
        config = MCPConfig.from_dict(config_dict)
        logger.info(f"MCP: Successfully loaded configuration with {len(config.servers)} server(s)")
        
        # Log enabled vs disabled servers
        enabled_count = len(config.get_enabled_servers())
        disabled_count = len(config.servers) - enabled_count
        logger.info(f"MCP: {enabled_count} enabled, {disabled_count} disabled server(s)")
        
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"MCP: Failed to parse JSON in {config_path}: {e}")
        logger.error(f"MCP: Line {e.lineno}, Column {e.colno}: {e.msg}")
        logger.error(f"MCP: Please check the JSON syntax in your configuration file")
        return None
    except PermissionError as e:
        logger.error(f"MCP: Permission denied reading {config_path}: {e}")
        logger.error(f"MCP: Please check file permissions")
        return None
    except Exception as e:
        logger.error(f"MCP: Unexpected error loading configuration from {config_path}: {e}", exc_info=True)
        logger.error(f"MCP: Please check the configuration file and try again")
        return None


def save_default_config(config_path: str = "mcp.json") -> bool:
    """
    Create a default MCP configuration file with example servers
    
    Args:
        config_path: Path where to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    default_config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/path/to/allowed/files"
                ],
                "env": {},
                "disabled": True
            },
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-github"
                ],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
                },
                "disabled": True
            },
            "postgres": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-postgres",
                    "postgresql://localhost/mydb"
                ],
                "env": {},
                "disabled": True
            }
        }
    }
    
    try:
        path = Path(config_path)
        with open(path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"MCP: Created default configuration at {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"MCP: Failed to create default configuration: {e}")
        return False


# Convenience function to get enabled servers directly
def get_enabled_servers(config_path: str = "mcp.json") -> List[MCPServerConfig]:
    """
    Load configuration and return list of enabled servers
    
    Args:
        config_path: Path to mcp.json configuration file
        
    Returns:
        List of enabled MCPServerConfig objects
    """
    config = load_config(config_path)
    if config is None:
        return []
    return config.get_enabled_servers()
