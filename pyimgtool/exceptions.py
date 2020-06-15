"""Custom exceptions for pyimgtool."""


class BaseException(Exception):
    """Common configuration for custom exceptions."""

    def __str__(self):
        return self.message


class ImageTooSmallError(BaseException):
    """Raised when the supplied image does not fit the intial size requirements."""

    def __init__(self, actual_size, required_size):
        """Pass in variables for exception message."""
        self.message = (
            f"Image is too small, Image size: {actual_size},"
            + f" Required size: {required_size}"
        )
        self.actual_size = actual_size
        self.required_size = required_size


class ResizeNotNeededError(BaseException):
    """Raised when current image size matches new size."""

    def __init__(self):
        self.message = "Resize not needed; image size matches new size."


class ResizeAttributeError(BaseException):
    """Raised when configuration is not correct for operation requested."""

    def __init__(self, message):
        self.message = message


class OverlaySizeError(BaseException):
    """Raised when one image can't be overlaid onto another."""

    def __init__(self, message):
        self.message = message
