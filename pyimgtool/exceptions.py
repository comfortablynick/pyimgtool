"""Custom exceptions for pyimgtool."""


class ImageTooSmallError(Exception):
    """Raised when the supplied image does not fit the intial size requirements."""

    def __init__(self, actual_size, required_size):
        """Pass in variables for exception message."""
        self.message = "Image is too small, Image size : %s, Required size : %s" % (
            actual_size,
            required_size,
        )
        self.actual_size = actual_size
        self.required_size = required_size

    def __str__(self):
        """Return string representation."""
        return self.message


class ResizeNotNeededError(Exception):
    """Raised when current image size matches new size."""

    def __init__(self):
        self.message = "Resize not needed; image size matches new size."

    def __str__(self):
        return self.message
