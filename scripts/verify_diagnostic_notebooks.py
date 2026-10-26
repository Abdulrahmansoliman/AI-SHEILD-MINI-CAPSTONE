"""Execute every cell twice against real downloaded tables; mock only Colab/UI.

Usage: python scripts/verify_diagnostic_notebooks.py <local-MyDrive-mirror>
Original Drive image files need not be downloaded: missing-image placeholders
are tested here, and verified-image rendering is covered by unit tests.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def verify(mirror):
    root = Path(mirror).resolve()
    # Real notebook analysis executes unchanged; only Drive mounting and UI display
    # are substituted, and its root points at a local byte-for-byte input mirror.
    google = types.ModuleType('google')
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=lambda path: None)
    google.colab = colab
    sys.modules['google'] = google
    sys.modules['google.colab'] = colab
    display_module = types.ModuleType('IPython.display')
    display_module.display = lambda value: None
    sys.modules['IPython'] = types.ModuleType('IPython')
    sys.modules['IPython.display'] = display_module
    records = []
    for notebook in sorted((ROOT / 'notebooks').glob('AI_Shield_0[123]_*_v1.ipynb')):
        payload = json.loads(notebook.read_text(encoding='utf-8'))
        for attempt in (1, 2):
            namespace = {'__name__': '__main__'}
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                for i, cell in enumerate(payload['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source']).replace(
                        "DRIVE_ROOT = Path('/content/drive/MyDrive')", f'DRIVE_ROOT = Path({str(root)!r})')
                    exec(compile(source, f'{notebook.name}:cell{i}', 'exec'), namespace)
            records.append({'notebook': notebook.name, 'attempt': attempt,
                'status': 'pass', 'output_directory': str(namespace['OUT']),
                'code_cells': sum(c['cell_type'] == 'code' for c in payload['cells'])})
            print(notebook.name, 'PASS', attempt, flush=True)
            log = root.parent / (notebook.stem + f'_attempt{attempt}.log')
            log.write_text(capture.getvalue(), encoding='utf-8')
    target = root.parent / 'notebook_execution_verification.json'
    target.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    verify(sys.argv[1])
