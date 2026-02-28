"""Model adapters for various inference backends."""

from apex.models.base import ModelAdapter, get_adapter

__all__ = ["ModelAdapter", "get_adapter"]
