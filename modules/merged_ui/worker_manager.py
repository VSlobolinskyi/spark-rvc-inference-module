import threading
import time
import logging
import os
from queue import Queue

from merged_ui.workers import persistent_rvc_worker, persistent_tts_worker

# Get model delay from environment variable or use default
DEFAULT_MODEL_DELAY = int(os.environ.get('MODEL_UNLOAD_DELAY', 60))

class WorkerManger:
    """
    Manages persistent TTS and RVC model instances with configurable unload delay.
    """
    def __init__(self, unload_delay=DEFAULT_MODEL_DELAY):
        """
        Initialize the model manager.
        
        Args:
            unload_delay (int): Time in seconds to keep models loaded after processing completes
        """
        self.unload_delay = unload_delay
        self.tts_workers = {}  # {worker_id: {'thread': thread, 'model': model, 'last_used': timestamp}}
        self.rvc_workers = {}  # {worker_id: {'thread': thread, 'last_used': timestamp}}
        self.tts_job_queues = {}  # {worker_id: Queue()}
        self.rvc_job_queues = {}  # {worker_id: Queue()}
        self.tts_active = {}  # {worker_id: Event()}
        self.rvc_active = {}  # {worker_id: Event()}
        self.manager_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Start unload monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_workers)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logging.info(f"Model Manager initialized with {unload_delay}s unload delay")
    
    def _monitor_workers(self):
        """Background thread that monitors worker usage and unloads idle workers."""
        while not self.shutdown_event.is_set():
            current_time = time.time()
            
            with self.manager_lock:
                # Check TTS workers
                for worker_id in list(self.tts_workers.keys()):
                    worker_info = self.tts_workers[worker_id]
                    if not self.tts_active[worker_id].is_set() and \
                       current_time - worker_info['last_used'] > self.unload_delay:
                        logging.info(f"Unloading idle TTS worker {worker_id} after {self.unload_delay}s")
                        # Signal the worker to shut down
                        self.tts_job_queues[worker_id].put(None)
                        # Wait for the worker to finish
                        worker_info['thread'].join(timeout=5)
                        # Clean up
                        del self.tts_workers[worker_id]
                        del self.tts_job_queues[worker_id]
                        del self.tts_active[worker_id]
                
                # Check RVC workers
                for worker_id in list(self.rvc_workers.keys()):
                    worker_info = self.rvc_workers[worker_id]
                    if not self.rvc_active[worker_id].is_set() and \
                       current_time - worker_info['last_used'] > self.unload_delay:
                        logging.info(f"Unloading idle RVC worker {worker_id} after {self.unload_delay}s")
                        # Signal the worker to shut down
                        self.rvc_job_queues[worker_id].put(None)
                        # Wait for the worker to finish
                        worker_info['thread'].join(timeout=5)
                        # Clean up
                        del self.rvc_workers[worker_id]
                        del self.rvc_job_queues[worker_id]
                        del self.rvc_active[worker_id]
            
            # Check every second
            time.sleep(1)
    
    def get_tts_worker(self, worker_id, model_dir, device):
        """
        Get or create a TTS worker.
        
        Args:
            worker_id (int): Unique worker ID
            model_dir (str): Path to TTS model directory
            device (str): Device to use for inference
            
        Returns:
            Queue: Job queue for the worker
        """
        with self.manager_lock:
            if worker_id in self.tts_workers:
                logging.info(f"Reusing existing TTS worker {worker_id}")
                self.tts_workers[worker_id]['last_used'] = time.time()
                self.tts_active[worker_id].set()
                return self.tts_job_queues[worker_id]
            
            # Create new worker
            logging.info(f"Creating new persistent TTS worker {worker_id}")
            job_queue = Queue()
            active_event = threading.Event()
            active_event.set()  # Mark as active initially
            
            self.tts_job_queues[worker_id] = job_queue
            self.tts_active[worker_id] = active_event
            
            thread = threading.Thread(
                target=persistent_tts_worker,
                args=(worker_id, job_queue, active_event, model_dir, device, self)
            )
            thread.daemon = True
            thread.start()
            
            self.tts_workers[worker_id] = {
                'thread': thread,
                'last_used': time.time()
            }
            
            return job_queue
    
    def get_rvc_worker(self, worker_id, cuda_stream=None):
        """
        Get or create an RVC worker.
        
        Args:
            worker_id (int): Unique worker ID
            cuda_stream: CUDA stream to use
            
        Returns:
            Queue: Job queue for the worker
        """
        with self.manager_lock:
            if worker_id in self.rvc_workers:
                logging.info(f"Reusing existing RVC worker {worker_id}")
                self.rvc_workers[worker_id]['last_used'] = time.time()
                self.rvc_active[worker_id].set()
                return self.rvc_job_queues[worker_id]
            
            # Create new worker
            logging.info(f"Creating new persistent RVC worker {worker_id}")
            job_queue = Queue()
            active_event = threading.Event()
            active_event.set()  # Mark as active initially
            
            self.rvc_job_queues[worker_id] = job_queue
            self.rvc_active[worker_id] = active_event
            
            thread = threading.Thread(
                target=persistent_rvc_worker,
                args=(worker_id, cuda_stream, job_queue, active_event, self)
            )
            thread.daemon = True
            thread.start()
            
            self.rvc_workers[worker_id] = {
                'thread': thread,
                'last_used': time.time()
            }
            
            return job_queue
    
    def mark_tts_worker_idle(self, worker_id):
        """Mark a TTS worker as idle."""
        with self.manager_lock:
            if worker_id in self.tts_workers:
                self.tts_workers[worker_id]['last_used'] = time.time()
                self.tts_active[worker_id].clear()
    
    def mark_rvc_worker_idle(self, worker_id):
        """Mark an RVC worker as idle."""
        with self.manager_lock:
            if worker_id in self.rvc_workers:
                self.rvc_workers[worker_id]['last_used'] = time.time()
                self.rvc_active[worker_id].clear()
    
    def update_unload_delay(self, delay):
        """Update the unload delay."""
        with self.manager_lock:
            self.unload_delay = delay
            logging.info(f"Updated model unload delay to {delay}s")
    
    def shutdown(self):
        """Shut down all workers and the manager."""
        logging.info("Shutting down Model Manager")
        self.shutdown_event.set()
        
        with self.manager_lock:
            # Signal all workers to shut down
            for queue in self.tts_job_queues.values():
                queue.put(None)
            for queue in self.rvc_job_queues.values():
                queue.put(None)
            
            # Wait for workers to finish
            for worker_info in self.tts_workers.values():
                worker_info['thread'].join(timeout=5)
            for worker_info in self.rvc_workers.values():
                worker_info['thread'].join(timeout=5)
        
        self.monitor_thread.join(timeout=5)
        logging.info("Model Manager shutdown complete")


# Global model manager instance
_model_manager = None

def get_worker_manager(unload_delay=None):
    """
    Get or create the global model manager.
    
    Args:
        unload_delay (int, optional): Time in seconds to keep models loaded after processing.
                                     If None, uses the current delay or default.
    
    Returns:
        WorkerManger: The global model manager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = WorkerManger(unload_delay or DEFAULT_MODEL_DELAY)
    elif unload_delay is not None and unload_delay != _model_manager.unload_delay:
        _model_manager.update_unload_delay(unload_delay)
    return _model_manager


# Create functions for UI integration
def set_worker_unload_delay(delay):
    """
    Set the model unload delay. Can be called directly from UI components.
    
    Args:
        delay (int): New delay in seconds
    
    Returns:
        str: Confirmation message
    """
    if not isinstance(delay, int) or delay < 0:
        return "Invalid delay value. Please provide a positive integer."
    
    get_worker_manager(delay)
    return f"Model unload delay set to {delay} seconds"


def get_current_worker_unload_delay():
    """
    Get the current model unload delay.
    
    Returns:
        int: Current delay in seconds
    """
    manager = get_worker_manager()
    return manager.unload_delay