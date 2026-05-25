"""Pipeline-specific exception hierarchy."""


class PipelineError(Exception):
    """Base class for all data-pipeline domain errors."""


class ConfigurationError(PipelineError):
    """Raised when required runtime configuration is missing or invalid."""


class ProviderRequestError(PipelineError):
    """Raised for provider/API request failures."""


class PersistenceError(PipelineError):
    """Raised for persistence failures to downstream stores."""


class RateLimitError(PipelineError):
    """Raised when rate-limit control cannot acquire a slot in time."""


class NotificationError(PipelineError):
    """Raised for notification publishing failures."""
