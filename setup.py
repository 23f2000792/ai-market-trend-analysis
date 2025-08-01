#!/usr/bin/env python3
"""
AI Market Trend Analysis - Setup Script
======================================

This script helps beginners set up the project quickly by:
1. Creating necessary directories
2. Running the complete pipeline
3. Providing helpful guidance

Author: AI Assistant
Created for: Market Trend Analysis Project
"""

import os
import sys
import subprocess
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command with error handling."""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ Error in {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data",
        "data/raw",
        "data/features",
        "models",
        "notebooks",
        "streamlit_app"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'yfinance',
        'streamlit', 'plotly', 'joblib'
    ]


    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n🎉 All required packages are installed!")
    return True

def main():
    """Main setup function."""

    print_header("AI Market Trend Analysis - Setup Script")
    print("Welcome! This script will set up your AI market analysis project.")
    print("Estimated time: 5-10 minutes (depending on your internet speed)")

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return

    print(f"✅ Python version: {python_version.major}.{python_version.minor}")

    # Step 1: Create directories
    print_step(1, "Creating Project Structure")
    create_directories()

    # Step 2: Check dependencies
    print_step(2, "Checking Dependencies")
    if not check_dependencies():
        print("\n🚨 Please install missing packages and run this script again.")
        return

    # Step 3: Collect data
    print_step(3, "Collecting Stock Market Data")
    print("This will download 5 years of data for AAPL, GOOGL, MSFT, AMZN, TSLA...")
    if run_command("python data_collector.py", "Data collection"):
        print("💾 Raw data saved to: data/raw/stock_data.csv")
    else:
        print("⚠️  Data collection failed. You can try running 'python data_collector.py' manually later.")

    # Step 4: Engineer features
    print_step(4, "Engineering Technical Features")
    print("Creating 40+ technical indicators...")
    if run_command("python feature_engineer.py", "Feature engineering"):
        print("🔧 Processed data saved to: data/features/stock_features.csv")
    else:
        print("⚠️  Feature engineering failed. You can try running 'python feature_engineer.py' manually later.")

    # Step 5: Train models
    print_step(5, "Training Machine Learning Models")
    print("Training Random Forest, Logistic Regression, and XGBoost...")
    if run_command("python model_trainer.py", "Model training"):
        print("🤖 Trained models saved to: models/")
    else:
        print("⚠️  Model training failed. You can try running 'python model_trainer.py' manually later.")

    # Final instructions
    print_header("Setup Complete! 🎉")

    print("Your AI Market Trend Analysis project is ready!")
    print("\nNext steps:")
    print("1. 📊 Launch the dashboard:")
    print("   streamlit run streamlit_app/app.py")
    print("\n2. 🔍 Explore the notebooks:")
    print("   jupyter notebook notebooks/")
    print("\n3. 📚 Read the documentation:")
    print("   Open README.md for detailed usage guide")

    print("\n🌐 Dashboard will be available at: http://localhost:8501")
    print("\n⚠️  Remember: This is for educational purposes only, not financial advice!")

    # Offer to launch dashboard
    try:
        launch = input("\n🚀 Would you like to launch the dashboard now? (y/n): ").lower().strip()
        if launch == 'y' or launch == 'yes':
            print("\n🌐 Launching Streamlit dashboard...")
            os.system("streamlit run streamlit_app/app.py")
    except KeyboardInterrupt:
        print("\n👋 Setup complete! Run the dashboard anytime with: streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()