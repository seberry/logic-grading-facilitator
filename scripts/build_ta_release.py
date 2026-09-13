"""Build and audit a code-only TA ZIP from committed files, not the worktree."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


def build(output_dir: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    short = commit[:7]
    roots = {'app', 'proofs_ocr', 'modal_ocr', 'scripts', 'docs', 'tests', 'LICENSES'}
    files = {'README.md', 'AGENTS.md', 'HOWTOUSE.md', 'requirements.txt',
             'requirements-lock.txt', 'pytest.ini', 'THIRD_PARTY_NOTICES.md',
             'VENDORED_COMPONENTS.md', 'START_INSTRUCTOR_GRADER.bat'}
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=repo, text=True).splitlines()
    selected = [name for name in tracked if name.split('/')[0] in roots or name in files]
    forbidden = {'.zip', '.pdf', '.jpg', '.jpeg', '.png', '.ico', '.tex'}
    selected = [name for name in selected if Path(name).suffix.lower() not in forbidden
                and Path(name).name != '.env']
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f'logic-grading-facilitator-{short}.zip'
    if archive.exists():
        raise RuntimeError(f'Refusing to overwrite {archive}')
    prefix = f'logic-grading-facilitator-{short}/'
    # Git preserves file modes and reads only the specified committed paths.
    subprocess.run(['git', 'archive', '--format=zip', f'--prefix={prefix}',
                    f'--output={archive.resolve()}', commit, '--', *selected], cwd=repo, check=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for name in names:
            relative = name.removeprefix(prefix)
            if not name.startswith(prefix) or relative.startswith(('data/', 'working/', '.git/')):
                raise RuntimeError(f'Unsafe archive path: {name}')
        for required in ('app/server.py', 'requirements.txt', 'requirements-lock.txt', 'docs/TA_GRADING_GUIDE.md'):
            if prefix + required not in names:
                raise RuntimeError(f'Missing release file: {required}')
        if bundle.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output_dir / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(f'Commit: {commit}\nArchive: {archive}\nSHA256: {digest}\nEntries: {len(names)}')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    build(parser.parse_args().output_dir)
