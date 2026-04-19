#!/usr/bin/env python3
"""
🚀 Dashboard Launch Script
User Behavior Analytics & A/B Testing Intelligence System

This script launches the Streamlit dashboard for the User Behavior Analytics system.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    # Get the project root directory
    project_root = Path(__file__).parent
    dashboard_path = project_root / "dashboard" / "app.py"
    
    # Check if the dashboard file exists
    if not dashboard_path.exists():
        print("❌ Error: Dashboard file not found!")
        print(f"Expected path: {dashboard_path}")
        return 1
    
    # Change to dashboard directory
    os.chdir(project_root / "dashboard")
    
    print("🚀 Launching User Behavior Analytics Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔗 If not opened automatically, visit: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
