import time
import logging
import os
import soundfile as sf

class AudioBufferQueue:
    """
    A buffer queue for audio file outputs that paces the release of files based on their duration.
    This ensures each audio file has time to finish playing before the next one is released.
    """
    def __init__(self, buffer_time):
        """
        Initialize the buffer queue.
        
        Args:
            buffer_time (float): Additional buffer time in seconds to add to each audio's playback
                                 to account for Gradio startup delay and ensure smooth transitions
        """
        self.queue = []  # Queue to store (file_path, duration) tuples
        self.current_file = None  # Currently playing file path
        self.current_duration = 0  # Duration of currently playing file in seconds
        self.playback_start_time = None  # When the current file started playing
        self.buffer_time = buffer_time  # Extra time to ensure complete playback
        
    def add(self, file_path):
        """
        Add a file to the buffer queue.
        
        Args:
            file_path (str): Path to the audio file
        """
        if file_path and os.path.exists(file_path):
            try:
                # Get audio duration using soundfile
                with sf.SoundFile(file_path) as sound_file:
                    duration = len(sound_file) / sound_file.samplerate
                
                # Add file to queue with its duration
                self.queue.append((file_path, duration))
                logging.info(f"Added file to buffer queue: {file_path} (duration: {duration:.2f}s)")
            except Exception as e:
                logging.error(f"Error getting audio duration: {str(e)}")
                # If we can't get duration, use a default value
                self.queue.append((file_path, 2.0))  # More conservative default
        else:
            # If file doesn't exist, add it with zero duration to pass through immediately
            self.queue.append((file_path, 0))
    
    def get_next(self):
        """
        Get the next file from the queue if enough time has passed for the current file.
        
        Returns:
            str or None: The file path if ready, None otherwise
        """
        current_time = time.time()
        
        # If we're currently playing a file, check if it's finished based on real elapsed time
        if self.current_file is not None and self.playback_start_time is not None:
            elapsed_time = current_time - self.playback_start_time
            
            # Effective playback time includes buffer time for Gradio initialization
            effective_duration = self.current_duration + self.buffer_time
            time_remaining = effective_duration - elapsed_time
            
            # Debugging information
            logging.debug(f"Current file: {self.current_file}, Elapsed: {elapsed_time:.2f}s, " +
                         f"Original Duration: {self.current_duration:.2f}s, " +
                         f"Effective Duration: {effective_duration:.2f}s, " +
                         f"Remaining: {time_remaining:.2f}s")
            
            # If not enough time has passed with the buffer
            if elapsed_time < effective_duration:
                return None
            
            # Log that the file has finished playing
            logging.info(f"Finished playing {self.current_file} " +
                         f"(duration: {self.current_duration:.2f}s, effective: {effective_duration:.2f}s)")
            self.current_file = None
        
        # If we don't have a current file and there are files in the queue, get the next one
        if self.current_file is None and self.queue:
            file_path, duration = self.queue.pop(0)
            self.current_file = file_path
            self.current_duration = duration
            self.playback_start_time = current_time
            
            # Calculate effective duration with buffer
            effective_duration = duration + self.buffer_time
            logging.info(f"Started playing {file_path} " + 
                         f"(duration: {duration:.2f}s, effective: {effective_duration:.2f}s)")
            return file_path
        
        return None
    
    def is_empty(self):
        """
        Check if the queue is empty and there's no current file.
        
        Returns:
            bool: True if empty, False otherwise
        """
        return len(self.queue) == 0 and self.current_file is None

    def clear(self):
        """
        Clear the queue and stop tracking the current file.
        Useful when wanting to immediately interrupt playback.
        """
        self.queue = []
        self.current_file = None
        self.playback_start_time = None
        logging.info("Audio buffer queue cleared")


class OrderedAudioBufferQueue(AudioBufferQueue):
    """
    Enhanced audio buffer queue that maintains sequential order regardless of when files are added.
    Files can be processed in parallel but will be played in the correct sequence.
    """
    def __init__(self, buffer_time=0.5):
        super().__init__(buffer_time)
        # Dictionary to store pending files by their position
        self.pending_files = {}  # {position: (file_path, duration)}
        self.next_position = 0  # Next position to output
        
    def add_with_position(self, file_path, position):
        """
        Add a file to the buffer queue with its position in the sequence.
        
        Args:
            file_path (str): Path to the audio file
            position (int): Position in the sequence (0-based)
        """
        if file_path and os.path.exists(file_path):
            try:
                # Get audio duration using soundfile
                with sf.SoundFile(file_path) as sound_file:
                    duration = len(sound_file) / sound_file.samplerate
                
                # Store the file with its position
                self.pending_files[position] = (file_path, duration)
                logging.info(f"Added file to ordered buffer queue: {file_path} at position {position} (duration: {duration:.2f}s)")
                
                # Check if we can add sequential files to the queue
                self._move_next_pending_to_queue()
            except Exception as e:
                logging.error(f"Error getting audio duration: {str(e)}")
                # If we can't get duration, use a default value
                self.pending_files[position] = (file_path, 2.0)  # Conservative default
                self._move_next_pending_to_queue()
        else:
            # If file doesn't exist, add it with zero duration
            logging.warning(f"File does not exist: {file_path}")
            self.pending_files[position] = (file_path, 0)
            self._move_next_pending_to_queue()
    
    def _move_next_pending_to_queue(self):
        """
        Move the next pending file (if available) to the queue.
        """
        while self.next_position in self.pending_files:
            file_path, duration = self.pending_files.pop(self.next_position)
            self.queue.append((file_path, duration))
            logging.info(f"Moved file from pending to queue: {file_path} at position {self.next_position}")
            self.next_position += 1
    
    def get_next(self):
        """
        Get the next file from the queue if enough time has passed for the current file.
        
        Returns:
            str or None: The file path if ready, None otherwise
        """
        # First check if current file has finished playing
        result = super().get_next()
        
        # If we just finished a file and might have pending files, check if we can move any to the queue
        if self.current_file is None and not result and self.pending_files:
            self._move_next_pending_to_queue()
            # Try again to get the next file
            result = super().get_next()
            
        return result
    
    def clear(self):
        """
        Clear the queue and stop tracking the current file.
        """
        super().clear()
        self.pending_files = {}
        self.next_position = 0