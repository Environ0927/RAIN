"""Explicit errors raised at protocol trust boundaries."""


class RainError(Exception):
    """Base class for RAIN simulator errors."""


class ValidationError(RainError, ValueError):
    """Base class for malformed public inputs."""


class ShareValidationError(ValidationError):
    """A Boolean share has an invalid shape, dtype, or bit value."""


class MessageValidationError(ValidationError):
    """A serialized protocol message is malformed or unexpected."""


class ProtocolError(RainError):
    """A protocol step was called out of order or with inconsistent state."""
