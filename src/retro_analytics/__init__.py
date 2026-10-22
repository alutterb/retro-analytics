"""Rank retro video games by forecast PriceCharting price growth."""


class PipelineError(Exception):
    """A stage could not do its job; the CLI reports it and exits non-zero."""
