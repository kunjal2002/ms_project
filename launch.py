#!/usr/bin/env python3
"""
Master Launcher for ResilientDB Educational MCP Server
Interactive guide to get everything set up and running
"""

import sys
import subprocess
import os
from pathlib import Path

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print welcome banner"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🎓 ResilientDB Educational MCP Server                      ║
║                                                                    ║
║        Master Setup & Launch Assistant                            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

def print_menu():
    """Print main menu"""
    print("\n📋 What would you like to do?\n")
    print("  1. 🧪 Verify Setup (Check if everything is ready)")
    print("  2. ⚙️  Configure Claude Desktop (Auto-setup)")
    print("  3. 🧪 Test All Components (Run test suite)")
    print("  4. 🚀 Start MCP Server (Manual test mode)")
    print("  5. 📚 View Documentation")
    print("  6. 🐳 Check Docker Status")
    print("  7. 📊 View Logs")
    print("  8. ❓ Help & Troubleshooting")
    print("  9. 🚪 Exit")
    print()

def run_verification():
    """Run verification script"""
    print("\n" + "="*70)
    print("🧪 Running Setup Verification...")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run([sys.executable, 'verify_setup.py'], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running verification: {e}")
        return False

def run_claude_config():
    """Run Claude Desktop configuration"""
    print("\n" + "="*70)
    print("⚙️  Configuring Claude Desktop...")
    print("="*70 + "\n")
    
    try:
        subprocess.run([sys.executable, 'setup_claude_desktop.py'], check=False)
        return True
    except Exception as e:
        print(f"❌ Error running configuration: {e}")
        return False

def run_tests():
    """Run component tests"""
    print("\n" + "="*70)
    print("🧪 Running Component Tests...")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run([sys.executable, 'test_educational_server.py'], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_server():
    """Start the MCP server manually"""
    print("\n" + "="*70)
    print("🚀 Starting MCP Server (Manual Mode)")
    print("="*70)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, 'educational_mcp_server.py'], check=False)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def view_docs():
    """Show documentation menu"""
    print("\n" + "="*70)
    print("📚 Documentation")
    print("="*70)
    print("\n  Available documentation files:\n")
    
    docs = [
        ("START_HERE.md", "Quick start guide - read this first!"),
        ("NEXT_STEPS.md", "What to do after setup"),
        ("QUICK_START.md", "5-minute getting started guide"),
        ("README_EDUCATIONAL.md", "Complete project documentation"),
        ("MS_PROJECT_DOCUMENTATION.md", "Technical deep dive"),
        ("PRESENTATION_OUTLINE.md", "MS project presentation"),
        ("DIFFERENTIATION.md", "How we're different"),
        ("CLAUDE_DESKTOP_SETUP.md", "Claude Desktop integration")
    ]
    
    for i, (filename, description) in enumerate(docs, 1):
        exists = "✅" if Path(filename).exists() else "❌"
        print(f"  {i}. {exists} {filename}")
        print(f"     {description}")
        print()
    
    print("\n💡 Tip: Open these files in your text editor or markdown viewer")

def check_docker():
    """Check Docker status"""
    print("\n" + "="*70)
    print("🐳 Docker Status")
    print("="*70 + "\n")
    
    try:
        # Check version
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✅ Docker installed: {result.stdout.strip()}")
            
            # Check if running
            result = subprocess.run(
                ['docker', 'ps'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                
                # Show running containers
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    print(f"\n📦 Running containers: {len(lines)-1}")
                    print(result.stdout)
                else:
                    print("\n📦 No containers running")
                
                print("\n✅ Interactive labs will work!")
            else:
                print("❌ Docker daemon not running")
                print("\n💡 Start Docker Desktop to enable interactive labs")
        else:
            print("❌ Docker not found")
            print("\n💡 Install Docker Desktop for interactive lab features")
            
    except FileNotFoundError:
        print("❌ Docker not installed")
        print("\n💡 Docker is optional but recommended for:")
        print("   - Interactive lab environments")
        print("   - Isolated student workspaces")
        print("   - Hands-on PBFT/GraphQL exercises")
        print("\nDownload from: https://www.docker.com/products/docker-desktop")
    except Exception as e:
        print(f"⚠️  Error checking Docker: {e}")

def view_logs():
    """View server logs"""
    print("\n" + "="*70)
    print("📊 Server Logs")
    print("="*70 + "\n")
    
    log_file = Path('educational_mcp.log')
    
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Show last 30 lines
            print("📄 Last 30 log entries:\n")
            print("-" * 70)
            for line in lines[-30:]:
                print(line.rstrip())
            print("-" * 70)
            
            print(f"\n📁 Full log file: {log_file.absolute()}")
            
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
    else:
        print("ℹ️  No log file found yet")
        print("   Logs will be created when you run the server")

def show_help():
    """Show help and troubleshooting"""
    print("\n" + "="*70)
    print("❓ Help & Troubleshooting")
    print("="*70 + "\n")
    
    print("🔍 Common Issues & Solutions:\n")
    
    issues = [
        ("Tools don't appear in Claude Desktop", [
            "1. Check config file is valid JSON",
            "2. Verify Python path in config",
            "3. Restart Claude Desktop completely",
            "4. Check logs: educational_mcp.log"
        ]),
        ("Module not found errors", [
            "1. Install dependencies: pip install -r requirements.txt",
            "2. Check Python version: python --version (need 3.8+)",
            "3. Verify in correct directory"
        ]),
        ("Docker errors in labs", [
            "1. Install Docker Desktop",
            "2. Start Docker Desktop",
            "3. Verify: docker ps",
            "4. Docker is optional - other features work without it"
        ]),
        ("Student data not saving", [
            "1. Check student_data/ directory exists",
            "2. Verify write permissions",
            "3. Check disk space"
        ])
    ]
    
    for issue, solutions in issues:
        print(f"❌ {issue}")
        for solution in solutions:
            print(f"   {solution}")
        print()
    
    print("\n📚 For more help:")
    print("   - Read: NEXT_STEPS.md")
    print("   - Check logs: educational_mcp.log")
    print("   - Review: README_EDUCATIONAL.md")

def show_quick_start():
    """Show quick start guide"""
    print("\n" + "="*70)
    print("🚀 Quick Start Guide")
    print("="*70 + "\n")
    
    print("Follow these steps to get started:\n")
    
    steps = [
        ("1. Verify Setup", "Run option 1 to check everything is ready"),
        ("2. Configure Claude", "Run option 2 to set up Claude Desktop"),
        ("3. Restart Claude Desktop", "Close and reopen Claude Desktop"),
        ("4. Test in Claude", 'Say: "I want to learn about BFT. My name is [Name]"'),
        ("5. Check Tools", "Look for 🔨 icon in Claude Desktop"),
        ("6. Start Learning", "Claude will use the educational tools automatically!")
    ]
    
    for step, description in steps:
        print(f"  ✓ {step}")
        print(f"    {description}\n")
    
    print("💡 See START_HERE.md for detailed instructions")

def main():
    """Main menu loop"""
    while True:
        clear_screen()
        print_banner()
        
        # Show quick status
        config_file = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        docker_available = subprocess.run(['docker', '--version'], capture_output=True).returncode == 0
        
        print("\n📊 Quick Status:")
        print(f"   {'✅' if config_file.exists() else '❌'} Claude Desktop Config")
        print(f"   {'✅' if docker_available else '⚠️ '} Docker Available")
        print(f"   {'✅' if Path('educational_mcp.log').exists() else 'ℹ️ '} Server Logs")
        
        print_menu()
        
        choice = input("👉 Select an option (1-9): ").strip()
        
        if choice == '1':
            run_verification()
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            run_claude_config()
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            run_tests()
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            start_server()
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            view_docs()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            check_docker()
            input("\nPress Enter to continue...")
        
        elif choice == '7':
            view_logs()
            input("\nPress Enter to continue...")
        
        elif choice == '8':
            show_help()
            input("\nPress Enter to continue...")
        
        elif choice == '9':
            print("\n👋 Goodbye! Happy learning!\n")
            break
        
        elif choice.lower() == 'quick':
            show_quick_start()
            input("\nPress Enter to continue...")
        
        else:
            print("\n❌ Invalid option. Please select 1-9.")
            input("Press Enter to try again...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("\nPress Enter to exit...")
