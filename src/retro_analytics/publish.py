"""Commit the export to the owner's private results repository.

Publishing is off until ``[publish]`` is set in ``settings.toml``. Before
anything is pushed, GitHub must confirm the repository is private:
PriceCharting's prices are for personal use, so a public or unverifiable
target is refused. Each run clones the branch into a throwaway directory
inside the data directory, replaces the file, and commits and pushes only when
the games, prices or thresholds changed.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from retro_analytics import PipelineError, paths, settings

GITHUB_URL = re.compile(
    r'^(?:git@github\.com:|ssh://git@github\.com/|https://github\.com/)'
    r'(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$')


class PublishError(PipelineError):
    """The export could not be published; the local file is still in place."""


def github_repo(url):
    """'owner/name' for a GitHub repository URL, else None."""
    match = GITHUB_URL.match(url.strip())
    return match['repo'] if match else None


def _last_line(stderr):
    return (stderr.strip().splitlines() or ['no output'])[-1]


def check_private(url):
    """Raise unless GitHub reports the repository as private."""
    parts = urlsplit(url)
    if parts.password or (parts.scheme in ('http', 'https') and parts.username):
        raise PublishError("publish.repo must not contain credentials; use an SSH URL or gh's credential helper")
    repo = github_repo(url)
    if repo is None:
        raise PublishError(f"publish.repo {url} is not a GitHub repository; only GitHub repositories can be "
                           "checked for privacy, so publishing is refused")
    try:
        result = subprocess.run(['gh', 'repo', 'view', repo, '--json', 'isPrivate', '--jq', '.isPrivate'],
                                capture_output=True, text=True, timeout=60)
    except FileNotFoundError as e:
        raise PublishError(f"cannot confirm {repo} is private: the GitHub CLI (gh) is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise PublishError(f"cannot confirm {repo} is private: gh timed out") from e
    if result.returncode != 0:
        raise PublishError(f"cannot confirm {repo} is private: gh failed ({_last_line(result.stderr)})")
    if result.stdout.strip() != 'true':
        raise PublishError(f"refusing to publish to {repo}: it is not private")


def _git(*args, cwd=None):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PublishError(f"git {args[0]} failed: {_last_line(result.stderr)}")
    return result.stdout


def _only_generated_differs(old_path, new_document):
    try:
        old = json.loads(old_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    old.pop('generated', None)
    return old == {k: v for k, v in new_document.items() if k != 'generated'}


def push(target, source):
    """Publish `source` to the repository and branch in `target`."""
    check_private(target.repo)
    document = json.loads(source.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(dir=paths.data_dir(), prefix='publish-') as work:
        clone = Path(work) / 'repo'
        if _git('ls-remote', '--heads', target.repo, target.branch).strip():
            _git('clone', '--quiet', '--depth', '1', '--branch', target.branch, target.repo, str(clone))
        else:
            _git('init', '--quiet', '--initial-branch', target.branch, str(clone))
            _git('remote', 'add', 'origin', target.repo, cwd=clone)
        destination = (clone / target.path).resolve()
        if not destination.is_relative_to(clone.resolve()):
            raise PublishError(f"publish.path {target.path} must stay inside the repository")
        if destination.exists() and _only_generated_differs(destination, document):
            print(f"Results unchanged; nothing published to {target.repo}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        _git('add', target.path, cwd=clone)
        _git('commit', '--quiet', '-m', f"Update results {document.get('generated', '')}".strip(), cwd=clone)
        _git('push', '--quiet', 'origin', f'HEAD:{target.branch}', cwd=clone)
    print(f"Published {target.path} to {target.repo} ({target.branch})")


def main():
    target = settings.load().publish
    if target is None:
        print("Publishing not configured; add [publish] to settings.toml to send results to the phone")
        return
    push(target, paths.require(paths.export(), "export"))
