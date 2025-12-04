# utils/tensorboard_utils.py
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def launch_tensorboard(log_dir: Path, port: int = 6006, wait_seconds: int = 5):
    """Launch TensorBoard and open browser automatically."""
    
    def start_tb():
        try:
            # Wait for log directory to be created and have some data
            time.sleep(wait_seconds)
            
            # Start TensorBoard
            cmd = [
                "tensorboard", 
                "--logdir", str(log_dir), 
                "--port", str(port),
                "--reload_multifile", "true",
                "--bind_all"  # Allow external access if needed
            ]
            
            print(f"Starting TensorBoard on http://localhost:{port}")
            print(f"Log directory: {log_dir}")
            
            process = subprocess.Popen(cmd)
            
            # Wait for TensorBoard to initialize
            time.sleep(3)
            
            # Open browser
            webbrowser.open(f"http://localhost:{port}")
            
            return process
        except Exception as e:
            print(f"Failed to start TensorBoard: {e}")
            print("You can manually start TensorBoard with:")
            print(f"tensorboard --logdir {log_dir} --port {port}")
            return None
    
    # Start in background thread
    #thread = threading.Thread(target=start_tb, daemon=True)
    #thread.start()
    return start_tb()
    #return thread