"""
run.py - Entry point for the Nosh Recipe Chatbot.

Usage:
    streamlit run run.py
    
Or directly:
    python -m streamlit run run.py
"""
import subprocess
import sys


def main():
    """Launch the Streamlit chatbot application."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/main.py"],
        cwd="."
    )


if __name__ == "__main__":
    main()

