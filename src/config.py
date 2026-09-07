from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    n_embd: int = 256
    dropout: float = 0.1
    N: int = 6
    h: int = 8
    
    @property
    def n_blocks(self) -> int:
        return 4 * self.n_embd

    @classmethod
    def standard_config(cls, vocab_size: int) -> Config:
        """A larger config for a 'Big Learning Run'."""
        return cls(vocab_size=vocab_size, n_embd=256, N=6, h=8)

    @classmethod
    def test_config(cls, vocab_size: int) -> Config:
        """A tiny config for quick testing and debugging."""
        return cls(vocab_size=vocab_size, n_embd=64, N=2, h=4)

    @classmethod
    def from_name(cls, name: str, vocab_size: int) -> Config:
        """Instantiate a config by name ('standard' or 'test')."""
        if name.lower() == "standard":
            return cls.standard_config(vocab_size)
        elif name.lower() == "test":
            return cls.test_config(vocab_size)
        else:
            raise ValueError(f"Unknown config name: {name}. Use 'standard' or 'test'.")
