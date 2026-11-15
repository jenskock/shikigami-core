"""HAL errors."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend's vendor libraries are not importable (e.g. on a Mac)."""
