"""Shared failures for exact repository-index generation binding."""


class ExactIndexPreconditionError(RuntimeError):
    """The requested exact structural generation cannot be safely used."""
