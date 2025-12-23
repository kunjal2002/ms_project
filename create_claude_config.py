#!/usr/bin/env python3
"""
Claude Desktop Configuration Helper for ResilientDB Expert MCP Server
Configures Claude Desktop to use the streamable HTTP transport
"""

import json
import os
from pathlib import Path
import platform

def create_claude_config_streamable_http(port: int = 8000):
    """Create Claude Desktop configuration for ResilientDB Expert MCP with streamable HTTP"""
    
    print(f"🔧 Configuring Claude Desktop for ResilientDB Expert MCP Server")
    print(f"🌐 Port: {port}")
    print(f"📡 Transport: Streamable HTTP")
    print("=" * 60)
    
    # Claude Desktop config locations by OS
    config_locations = {
        "Windows": Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
        "Darwin": Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # macOS
        "Linux": Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    }
    
    # Detect current OS
    current_os = platform.system()
    config_path = config_locations.get(current_os)
    
    if not config_path:
        print(f"❌ Unsupported OS: {current_os}")
        print(f"💡 Manually add configuration to your Claude Desktop config file")
        show_manual_config(port)
        return
    
    print(f"🖥️  Operating System: {current_os}")
    print(f"📁 Config path: {config_path}")
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing configuration or create new one
    config = {"mcpServers": {}}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("✅ Existing config file loaded")
        except Exception as e:
            print(f"⚠️  Could not read existing config: {e}")
            print("🔧 Creating new configuration")
            config = {"mcpServers": {}}
    else:
        print("📝 Creating new config file")
    
    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Get current project path for Python server
    current_path = Path(__file__).parent.absolute()
    server_path = current_path / "resilientdb_expert_mcp_server.py"
    
    # Add ResilientDB Expert Assistant configuration
    # Use stdio mode for better compatibility with Claude Desktop
    config["mcpServers"]["ResilientDB Expert Assistant"] = {
        "command": "python",
        "args": [str(server_path)]
    }
    
    # Write updated configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ Claude Desktop configuration updated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to write config file: {e}")
        show_manual_config(port)
        return
    
    print_usage_instructions(port, config_path)

def show_manual_config(port: int):
    """Show manual configuration instructions"""
    print("\n📋 Manual Configuration:")
    print("Add this to your Claude Desktop config file:")
    print("-" * 50)
    
    config = {
        "mcpServers": {
            "ResilientDB Expert Assistant": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-remote", f"http://localhost:{port}/mcp/"]
            }
        }
    }
    
    print(json.dumps(config, indent=2))

def print_usage_instructions(port: int, config_path: Path):
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("🚀 Setup Complete!")
    print("=" * 60)
    
    print(f"\n📁 Configuration file: {config_path}")
    
    print("\n🔄 Next Steps:")
    print("1. The server will start automatically when Claude Desktop connects")
    print("2. Restart Claude Desktop to load the new configuration")
    print("3. In Claude Desktop, you can now ask questions like:")
    print('   • "Tell me about Debitable and its social consensus features"')
    print('   • "How does DraftRes achieve provably fair gaming?"')
    print('   • "Explain ResilientDB\'s PBFT consensus implementation"')
    print('   • "Compare Arrayán and Echo applications"')
    
    print("\n🌐 For HTTP mode (optional):")
    print(f"   python resilientdb_expert_mcp_server.py http {port}")
    print(f"   Server URL: http://localhost:{port}/mcp")
    
    print("\n🔧 Troubleshooting:")
    print("• Check Claude Desktop logs if connection fails")
    print("• Ensure Python and required packages are installed")
    print("• Verify the server path is correct in configuration")
    
    print(f"\n💡 Test manually: python {Path(__file__).parent / 'resilientdb_expert_mcp_server.py'}")

def main():
    """Main function"""
    import sys
    
    # Get port from command line, default to 8000
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid port '{sys.argv[1]}', using default 8000")
    
    create_claude_config_streamable_http(port)

if __name__ == "__main__":
    main()
