"""Build the source-only AWS deployment artifact, excluding credentials, snapshots and sessions."""
import sys
import zipfile
from pathlib import Path

backend = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1] if len(sys.argv) > 1 else backend / 'deploy' / 'backend.zip').resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in backend.rglob('*'):
        relative = path.relative_to(backend)
        if not path.is_file() or any(x in relative.parts for x in ('data','.venv','__pycache__','.pytest_cache','tests','deploy')):
            continue
        if path.name.startswith('.env') or path.suffix in ('.zip','.log','.sqlite3'):
            continue
        archive.write(path, 'burhan/backend/' + relative.as_posix())
    archive.write(backend.parent / 'assets' / 'catalogue.json','burhan/assets/catalogue.json')
print(destination)
