import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('BURHAN_DATA_DIR', ROOT / 'data')).resolve())
    provider: str = field(default_factory=lambda: os.getenv('SMART_INSPECTION_PROVIDER', 'codex'))
    codex_command: str = field(default_factory=lambda: os.getenv('CODEX_COMMAND', 'codex'))
    codex_model: str = field(default_factory=lambda: os.getenv('CODEX_MODEL', ''))
    aws_region: str = field(default_factory=lambda: os.getenv('AWS_REGION', 'us-east-1'))
    aws_model: str = field(default_factory=lambda: os.getenv('AWS_BEDROCK_MODEL_ID', 'us.amazon.nova-2-lite-v1:0'))
    timeout: float = field(default_factory=lambda: float(os.getenv('AGENT_TIMEOUT_SECONDS', '120')))
    min_interval: float = field(default_factory=lambda: float(os.getenv('MIN_TURN_INTERVAL_SECONDS', '10')))
    session_limit: int = field(default_factory=lambda: int(os.getenv('SESSION_TURN_LIMIT', '60')))
    daily_limit: int = field(default_factory=lambda: int(os.getenv('DAILY_TURN_LIMIT', '300')))
    origins: list[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'https://merhaba-api.kiyan.finance,https://localhost,http://localhost,http://127.0.0.1:4173,http://localhost:4173,http://127.0.0.1:5173,http://localhost:5173').split(','))
    catalogue: Path = field(default_factory=lambda: ROOT.parent / 'assets' / 'catalogue.json')
    research_catalogue: Path = field(default_factory=lambda: ROOT.parent / 'assets' / 'research-catalogue.json')

    def __post_init__(self):
        if self.provider not in ('codex', 'aws', 'off'):
            raise ValueError('SMART_INSPECTION_PROVIDER must be codex, aws, or off')
        if self.timeout < 1 or self.min_interval < 0 or self.session_limit < 1 or self.daily_limit < 1:
            raise ValueError('Invalid agent limits')


def load_settings():
    load_dotenv(ROOT / '.env')
    return Settings()
