#!/usr/bin/env python3
"""
Utility script to launch TensorBoard for NSFWDetect visualizations.
"""
import argparse
import subprocess
import webbrowser
import time
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for NSFWDetect")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory containing TensorBoard logs")
    parser.add_argument("--port", type=int, default=6006,
                        help="Port to serve TensorBoard on")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't automatically open browser")
    args = parser.parse_args()

    # Verify log directory exists
    log_dir = Path(args.logdir)
    if not log_dir.exists():
        print(f"Warning: Log directory {log_dir} does not exist.")
        create_dir = input("Create it? (y/n): ")
        if create_dir.lower() == 'y':
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {log_dir}")
        else:
            print("Exiting.")
            return

    # Print the command we're running
    cmd = ["tensorboard", f"--logdir={args.logdir}", f"--port={args.port}"]
    print(f"Running: {' '.join(cmd)}")

    # Start TensorBoard
    tensorboard_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for TensorBoard to initialize
    print("Starting TensorBoard...")
    time.sleep(3)

    # Check if TensorBoard process is still running
    if tensorboard_process.poll() is not None:
        # Process terminated, print error
        stdout, stderr = tensorboard_process.communicate()
        print("TensorBoard failed to start:")
        print(stderr)
        return

    # TensorBoard is running
    url = f"http://localhost:{args.port}/"
    print(f"TensorBoard running at: {url}")

    # Open browser if requested
    if not args.no_browser:
        print("Opening in browser...")
        webbrowser.open(url)

    print("Press Ctrl+C to stop TensorBoard")

    try:
        # Keep the script running until user interrupts
        while True:
            time.sleep(1)

            # Check if process is still running
            if tensorboard_process.poll() is not None:
                stdout, stderr = tensorboard_process.communicate()
                print("TensorBoard exited unexpectedly:")
                print(stderr)
                break

    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        tensorboard_process.terminate()
        print("TensorBoard stopped.")

if __name__ == "__main__":
    main()
