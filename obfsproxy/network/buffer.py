class Buffer(object):
    """
    A Buffer is a simple FIFO buffer. You write() stuff to it, and you
    read() them back. You can also peek() or drain() data.
    """

    def __init__(self, data=''):
        """
        Initialize a buffer with 'data'.
        """
        self.buffer = bytes(data)

    def read(self, n=-1):
        """
        Read and return 'n' bytes from the buffer.

        If 'n' is negative, read and return the whole buffer.
        If 'n' is larger than the size of the buffer, read and return
        the whole buffer.
        """

        if (n < 0) or (n > len(self.buffer)):
            the_whole_buffer = self.buffer
            self.buffer = bytes('')
            return the_whole_buffer

        data = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return data

    def write(self, data):
        """
        Append 'data' to the buffer.
        """
        self.buffer = self.buffer + data

    def peek(self, n=-1):
        """
        Return 'n' bytes from the buffer, without draining them.

        If 'n' is negative, return the whole buffer.
        If 'n' is larger than the size of the buffer, return the whole
        buffer.
        """

        if (n < 0) or (n > len(self.buffer)):
            return self.buffer

        return self.buffer[:n]

    def drain(self, n=-1):
        """
        Drain 'n' bytes from the buffer.

        If 'n' is negative, drain the whole buffer.
        If 'n' is larger than the size of the buffer, drain the whole
        buffer.
        """
        if (n < 0) or (n > len(self.buffer)):
            self.buffer = bytes('')
            return

        self.buffer = self.buffer[n:]
        return

    def __len__(self):
        """Returns length of buffer. Used in len()."""
        return len(self.buffer)

    def __nonzero__(self):
        """
        Returns True if the buffer is non-empty.
        Used in truth-value testing.
        """
        return True if len(self.buffer) else False
