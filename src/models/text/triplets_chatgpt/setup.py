#!/usr/bin/env python3
"""
Setup script for the Referring Expression Processor.
Makes it easy to get started with the system.
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment file."""
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    if os.path.exists(".env.template"):
        print("📝 Creating .env file from template...")
        with open(".env.template", "r") as template:
            content = template.read()
        
        with open(".env", "w") as env_file:
            env_file.write(content)
        
        print("✅ .env file created!")
        print("⚠️  Please add your Claude API key to the .env file")
        return True
    else:
        print("❌ .env.template not found!")
        return False

def check_files():
    """Check if all required files are present."""
    required_files = [
        "main_processor.py",
        "test_validator.py", 
        "test_cops_ref_data.json",
        "requirements.txt",
        ".env.template"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING!")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  No API key found in .env file")
        print("   Please add your Claude API key and run: python test_validator.py quick")
        return False
    
    try:
        subprocess.check_call([sys.executable, "test_validator.py", "quick"])
        return True
    except subprocess.CalledProcessError:
        print("❌ Quick test failed. Run 'python test_validator.py' for details.")
        return False

def show_next_steps():
    """Show user what to do next."""
    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("="*50)
    
    print("\n📋 Next Steps:")
    print("1. Add your Claude API key to .env file:")
    print("   ANTHROPIC_API_KEY=your_actual_key_here")
    print("\n2. Test the system:")
    print("   python test_validator.py quick")
    print("\n3. Process your data:")
    print("   python main_processor.py test_cops_ref_data.json output.jsonl")
    print("\n4. Check the results:")
    print("   head output.jsonl")
    
    print("\n🔗 Get Claude API Key:")
    print("   https://console.anthropic.com/")
    
    print("\n📚 Full Documentation:")
    print("   See README.md for complete usage instructions")

def main():
    """Main setup function."""
    print("🚀 Referring Expression Processor Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check required files
    print("\n📁 Checking required files...")
    if not check_files():
        print("❌ Setup failed: missing required files")
        return False
    
    # Install dependencies
    print("\n📦 Setting up dependencies...")
    if not install_dependencies():
        return False
    
    # Setup environment
    print("\n⚙️  Setting up environment...")
    if not setup_environment():
        return False
    
    # Run quick test if API key is available
    api_test_passed = run_quick_test()
    
    # Show next steps
    show_next_steps()
    
    return api_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
