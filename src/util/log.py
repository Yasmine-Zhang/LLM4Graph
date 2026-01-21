import logging
import sys
import os

# Custom handler for BlobFuse synchronization
# (Kept for compatibility with potential cloud storage mounts, as seen in reference code)
class DirectFileHandler(logging.Handler):
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        
        # Initialize file (truncate if mode is 'w')
        if mode == 'w':
             with open(self.filename, 'w', encoding=self.encoding) as f:
                pass
             self.mode = 'a' # Switch to append for subsequent writes

    def emit(self, record):
        try:
            msg = self.format(record)
            # Force Open-Write-Close for every log to ensure sync
            with open(self.filename, self.mode, encoding=self.encoding) as f:
                f.write(msg + '\n')
        except Exception:
            self.handleError(record)

def setup_logger(output_dir, log_file_name="run.log"):
    """
    Sets up a logger that writes to both stdout and a file in the output directory.
    Uses DirectFileHandler to ensure Immediate I/O consistency.
    """
    log_file_path = os.path.join(output_dir, log_file_name)
    
    # Remove existing handlers if any (to avoid duplicate logs in interactive sessions)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            DirectFileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Return both the filepath (for reference) and a logger instance
    logger = logging.getLogger(__name__)
    return logger, log_file_path
