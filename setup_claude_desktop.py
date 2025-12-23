#!/usr/bin/env python3
"""
Claude Desktop Configuration Helper for Educational MCP Server
Automatically configures Claude Desktop to use the ResilientDB Educational Assistant
"""

import json
import os
import sys
from pathlib import Path
import platform
import subprocess

def create_educational_claude_config():
    """Create Claude Desktop configuration for Educational MCP Server"""
    
    print("=" * 70)
    print("🎓 ResilientDB Educational Assistant - Claude Desktop Setup")
    print("=" * 70)
    
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
        show_manual_config()
        return False
    
    print(f"\n🖥️  Operating System: {current_os}")
    print(f"📁 Config path: {config_path}")
    
    # Check if Claude Desktop is installed
    if not config_path.parent.exists():
        print(f"\n⚠️  Warning: Claude Desktop config directory not found!")
        print(f"📂 Expected location: {config_path.parent}")
        print(f"\n💡 Make sure Claude Desktop is installed and has been run at least once.")
        print(f"   After running Claude Desktop, try this script again.")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
        
        # Create the directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing configuration or create new one
    config = {"mcpServers": {}}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("✅ Existing config file loaded")
                
                # Show existing servers
                if "mcpServers" in config and config["mcpServers"]:
                    print(f"📋 Found {len(config['mcpServers'])} existing MCP server(s):")
                    for server_name in config["mcpServers"].keys():
                        print(f"   - {server_name}")
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
    server_path = current_path / "educational_mcp_server.py"
    
    if not server_path.exists():
        print(f"\n❌ Error: educational_mcp_server.py not found at {server_path}")
        return False
    
    # Check Python availability
    python_cmd = sys.executable  # Use the same Python that's running this script
    print(f"\n🐍 Python executable: {python_cmd}")
    
    try:
        result = subprocess.run([python_cmd, "--version"], capture_output=True, text=True)
        print(f"✅ Python version: {result.stdout.strip()}")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify Python: {e}")
    
    # Add Educational Assistant configuration (stdio mode)
    server_name = "resilientdb-education"
    
    if server_name in config["mcpServers"]:
        print(f"\n⚠️  Server '{server_name}' already exists in config")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("❌ Configuration cancelled")
            return False
    
    config["mcpServers"][server_name] = {
        "command": python_cmd,
        "args": [str(server_path)],
        "env": {}
    }
    
    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix('.json.backup')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            print(f"💾 Backup created: {backup_path}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
    
    # Write updated configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("✅ Claude Desktop configuration updated successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Failed to write config file: {e}")
        show_manual_config()
        return False
    
    print_next_steps(config_path)
    return True

def show_manual_config():
    """Show manual configuration instructions"""
    current_path = Path(__file__).parent.absolute()
    server_path = current_path / "educational_mcp_server.py"
    python_cmd = sys.executable
    
    print("\n" + "=" * 70)
    print("📋 MANUAL CONFIGURATION INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Locate your Claude Desktop config file:")
    print("   Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
    print("   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("   Linux: ~/.config/claude/claude_desktop_config.json")
    
    print("\n2. Add this configuration to the file:")
    print("-" * 70)
    
    config = {
        "mcpServers": {
            "resilientdb-education": {
                "command": python_cmd,
                "args": [str(server_path)],
                "env": {}
            }
        }
    }
    
    print(json.dumps(config, indent=2))
    print("-" * 70)
    
    print("\n3. If you already have other MCP servers, merge this configuration")
    print("   with your existing mcpServers section.")

def print_next_steps(config_path):
    """Print next steps for the user"""
    print("\n📋 NEXT STEPS:")
    print("-" * 70)
    print("1. ✅ Configuration file updated")
    print(f"   Location: {config_path}")
    
    print("\n2. 🔄 Restart Claude Desktop")
    print("   - Close Claude Desktop completely")
    print("   - Reopen it")
    
    print("\n3. 🔍 Verify the server is connected:")
    print("   - Look for the 🔨 hammer icon in Claude Desktop")
    print("   - Click it to see available tools")
    print("   - You should see tools like:")
    print("     • start_learning_session")
    print("     • explain_concept")
    print("     • view_knowledge_graph")
    print("     • setup_resilientdb_environment")
    print("     • create_interactive_lab")
    print("     • ... and more!")
    
    print("\n4. 🎓 Start a learning session:")
    print('   Try this in Claude Desktop:')
    print('   "I want to learn about Byzantine Fault Tolerance.')
    print('    My name is [YourName] and I\'m a beginner."')
    
    print("\n5. 🐛 Troubleshooting (if tools don't appear):")
    print("   - Check the log file: educational_mcp.log")
    print("   - Ensure Python is in your PATH")
    print("   - Verify the config file syntax is valid JSON")
    print("   - Check Claude Desktop logs:")
    if platform.system() == "Windows":
        print("     %APPDATA%\\Claude\\logs\\")
    else:
        print("     ~/Library/Logs/Claude/ (macOS)")
    
    print("\n6. 📚 Documentation:")
    print("   - Quick Start: QUICK_START.md")
    print("   - Full Guide: README_EDUCATIONAL.md")
    print("   - Next Steps: NEXT_STEPS.md")
    
    print("\n" + "=" * 70)
    print("🎉 Setup complete! Happy learning!")
    print("=" * 70)

def verify_dependencies():
    """Verify required dependencies are installed"""
    print("\n🔍 Verifying dependencies...")
    
    required_packages = [
        'mcp',
        'docker',
        'gitpython',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Warning: Missing {len(missing)} required package(s)")
        print("Run this command to install them:")
        print(f"   pip install -r requirements.txt")
        
        response = input("\nInstall missing packages now? (y/n): ")
        if response.lower() == 'y':
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                print("✅ Dependencies installed successfully!")
            except Exception as e:
                print(f"❌ Failed to install dependencies: {e}")
                return False
    
    return True

def main():
    """Main entry point"""
    print("\n")
    
    # Verify dependencies first
    if not verify_dependencies():
        print("\n❌ Please install dependencies before continuing")
        return
    
    # Create configuration
    success = create_educational_claude_config()
    
    if success:
        print("\n✅ All done! You can now use the Educational MCP Server in Claude Desktop.")
    else:
        print("\n⚠️  Configuration incomplete. Please follow manual instructions above.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
