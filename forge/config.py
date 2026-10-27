from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("FORGE_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    max_tokens: int = int(os.getenv("FORGE_MAX_TOKENS", "512"))


settings = Settings()
