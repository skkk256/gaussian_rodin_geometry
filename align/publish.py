"""Build the local eight-layer viewer from alignment outputs."""
import json
import os
from pathlib import Path
from urllib.parse import quote

from io_utils import sha256, write_json
from . import PROJECT


def publish(output: Path) -> None:
    template = Path(__file__).parent/'viewer'
    dependencies = os.path.relpath(PROJECT/'third_party', output)
    html = (template/'index.html').read_text().replace('@THIRD_PARTY@', dependencies)
    (output/'index.html').write_text(html)
    for name in ('viewer.js', 'overlay.js'):
        (output/name).write_bytes((template/name).read_bytes())
    rows = []
    for folder in sorted(output.glob('*/*')):
        if not folder.is_dir():
            continue
        result, error = folder/'result.json', folder/'error.json'
        if not result.is_file() and not error.is_file():
            continue
        report = json.loads((result if result.is_file() else error).read_text())
        row = dict(case='/'.join(folder.relative_to(output).parts), status=report['status'],
                   base='/'.join(quote(part, safe='') for part in folder.relative_to(output).parts)+'/')
        if report['status'] == 'completed':
            row.update(validation=report['validation'], fit=report['fit'], totalSeconds=report['totalSeconds'],
                       accepted=report['diagnosticCheck']['accepted'], finalStage=report['finalStage'],
                       renderVersion=sha256(result)[:16])
        else:
            row['error'] = report['error']
        rows.append(row)
    write_json(output/'manifest.json', rows)
