import os
import sys
import logging
import io

logger = logging.getLogger(__name__)

class StderrSuppressor:
    """
    Context manager to suppress FFmpeg warnings and other unnecessary output.
    Captures stderr and only prints lines that don't contain filtered phrases.
    """
    def __enter__(self):
        # Create a text stream to capture stderr
        self.stderr_capture = io.StringIO()
        # Redirect stderr to our capture
        self.old_stderr = sys.stderr
        sys.stderr = self.stderr_capture
        return self.stderr_capture
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stderr
        sys.stderr = self.old_stderr
        # Filter out all deprecated pixel format warnings completely
        # Only print lines that don't contain our filtered phrases
        content = self.stderr_capture.getvalue()
        for line in content.splitlines():
            if "deprecated pixel format" not in line and "swscaler" not in line and line.strip():
                print(line, file=self.old_stderr)

# Create an instance of StderrSuppressor that can be imported and used as a context manager
ffmpeg_suppressor = StderrSuppressor()