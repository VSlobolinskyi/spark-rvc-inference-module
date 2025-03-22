import time
import logging
import os
import soundfile as sf

class AudioBufferQueue:
    """
    A combined audio buffer queue that supports both unordered and ordered addition of audio files.
    Files are buffered to pace playback based on their duration. When adding files with an ordering,
    use the 'position' parameter.
    """
    def __init__(self, buffer_time=1.0, ordered=False):
        """
        Initialize the buffer queue.
        
        Args:
            buffer_time (float): Extra time in seconds added to each audio's duration to ensure smooth transitions.
            ordered (bool): Whether to enable ordering support (pending files added with positions).
        """
        self.buffer_time = buffer_time
        self.queue = []  # List of (file_path, duration) tuples.
        self.current_file = None  # Currently playing file path.
        self.current_duration = 0  # Duration of the currently playing file.
        self.playback_start_time = None  # When the current file started playing.
        self.min_playback_time = 1.0  # Minimum time to keep an audio playing.
        
        # For ordered playback.
        self.ordered = ordered
        if self.ordered:
            self.pending_files = {}  # {position: (file_path, duration)}
            self.next_position = 0   # Next expected position for playback.

    def add(self, file_path, position=None):
        """
        Add a file to the buffer.
        
        If 'position' is provided, the file is added in an ordered manner.
        Otherwise, the file is appended to the playback queue.
        
        Args:
            file_path (str): Path to the audio file.
            position (int, optional): Position in the sequence (0-based) for ordered playback.
        """
        if position is not None:
            # Enable ordered mode if not already enabled.
            if not self.ordered:
                self.ordered = True
                self.pending_files = {}
                self.next_position = 0
            self._add_with_position(file_path, position)
        else:
            self._add_to_queue(file_path)

    def _add_to_queue(self, file_path):
        """
        Add the file directly to the playback queue.
        """
        if file_path and os.path.exists(file_path):
            try:
                with sf.SoundFile(file_path) as sound_file:
                    duration = len(sound_file) / sound_file.samplerate
                self.queue.append((file_path, duration))
                logging.info(f"Added file to buffer queue: {file_path} (duration: {duration:.2f}s)")
            except Exception as e:
                logging.error(f"Error getting audio duration for {file_path}: {str(e)}")
                self.queue.append((file_path, 2.0))  # Use conservative default.
        else:
            # If file does not exist, add it with zero duration.
            self.queue.append((file_path, 0))

    def _add_with_position(self, file_path, position):
        """
        Add the file with its position into the pending dictionary, then try moving pending files to the queue.
        """
        if file_path and os.path.exists(file_path):
            try:
                with sf.SoundFile(file_path) as sound_file:
                    duration = len(sound_file) / sound_file.samplerate
                self.pending_files[position] = (file_path, duration)
                logging.info(f"Added file to ordered queue: {file_path} at position {position} (duration: {duration:.2f}s)")
            except Exception as e:
                logging.error(f"Error getting audio duration for {file_path}: {str(e)}")
                self.pending_files[position] = (file_path, 2.0)
        else:
            logging.warning(f"File does not exist: {file_path}")
            self.pending_files[position] = (file_path, 0)
        self._move_next_pending_to_queue()

    def _move_next_pending_to_queue(self):
        """
        Move pending files into the playback queue in sequential order if available.
        """
        if self.ordered:
            while self.next_position in self.pending_files:
                file_path, duration = self.pending_files.pop(self.next_position)
                self.queue.append((file_path, duration))
                logging.info(f"Moved file from pending to queue: {file_path} at position {self.next_position}")
                self.next_position += 1

    def get_next(self):
        """
        Return the next file from the queue if enough time has passed for the current file.
        
        Returns:
            str or None: The file path if ready for playback; otherwise, None.
        """
        current_time = time.time()
        
        # Check if a file is currently playing and if its effective playback time has elapsed.
        if self.current_file is not None and self.playback_start_time is not None:
            elapsed_time = current_time - self.playback_start_time
            effective_duration = max(self.current_duration + self.buffer_time, self.min_playback_time)
            logging.debug(f"Current file: {self.current_file}, Elapsed: {elapsed_time:.2f}s, "
                          f"Original Duration: {self.current_duration:.2f}s, "
                          f"Effective Duration: {effective_duration:.2f}s")
            if elapsed_time < effective_duration:
                return None
            logging.info(f"Finished playing {self.current_file} (duration: {self.current_duration:.2f}s, effective: {effective_duration:.2f}s)")
            self.current_file = None

        # If ordered mode is enabled and there are pending files, try moving them into the queue.
        if self.ordered and self.pending_files:
            self._move_next_pending_to_queue()
        
        # If no file is currently playing and the queue is not empty, start the next file.
        if self.current_file is None and self.queue:
            file_path, duration = self.queue.pop(0)
            self.current_file = file_path
            self.current_duration = duration
            self.playback_start_time = current_time
            effective_duration = max(duration + self.buffer_time, self.min_playback_time)
            logging.info(f"Started playing {file_path} (duration: {duration:.2f}s, effective: {effective_duration:.2f}s)")
            return file_path
        
        return None

    def is_empty(self):
        """
        Check if both the playback queue is empty and no file is currently playing.
        
        Returns:
            bool: True if empty; otherwise, False.
        """
        return len(self.queue) == 0 and self.current_file is None

    def clear(self):
        """
        Clear the playback queue and any pending files, and reset current playback.
        """
        self.queue = []
        self.current_file = None
        self.playback_start_time = None
        logging.info("Audio buffer queue cleared")
        if self.ordered:
            self.pending_files = {}
            self.next_position = 0