"""Provider adapters for the PromptForge model-in-loop benchmark."""

from .openai_compatible import OpenAICompatibleAgentAdapter, OpenAICompatibleConfig

__all__ = ["OpenAICompatibleAgentAdapter", "OpenAICompatibleConfig"]
