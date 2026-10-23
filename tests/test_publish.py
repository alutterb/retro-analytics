import json
import subprocess

import pytest

from retro_analytics import cli, export, paths, publish

TARGET = "git@github.com:alutterb/retro-analytics-data.git"


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    for var in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{var}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{var}_EMAIL", "test@example.com")


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """An empty bare repository standing in for the private results repo."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    monkeypatch.setattr(publish, "check_private", lambda repo: None)
    paths.settings().write_text(f'[publish]\nrepo = "{bare}"\n', encoding="utf-8")
    return bare


def write_export(games=(), generated="2026-10-25"):
    export.write({"schema_version": 1, "generated": generated, "games": list(games)}, paths.export())


def remote_log(bare):
    result = subprocess.run(["git", "-C", str(bare), "log", "--format=%s", "main"],
                            capture_output=True, text=True)
    return result.stdout.split("\n")[:-1] if result.returncode == 0 else []


def remote_file(bare, path="results/latest.json"):
    return json.loads(subprocess.run(["git", "-C", str(bare), "show", f"main:{path}"],
                                     capture_output=True, text=True, check=True).stdout)


def fake_gh(tmp_path, monkeypatch, script):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\n" + script + "\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


def test_unset_publishing_is_a_no_op(monkeypatch, capsys):
    # Covers AE4.
    monkeypatch.setattr(publish, "_git", lambda *args, **kwargs: pytest.fail("git called"))
    publish.main()
    assert "not configured" in capsys.readouterr().out


def test_first_publish_creates_the_branch_on_an_empty_remote(remote):
    write_export([{"game": "zelda"}])
    publish.main()
    assert remote_log(remote) == ["Update results 2026-10-25"]
    assert remote_file(remote)["games"] == [{"game": "zelda"}]


def test_identical_file_makes_no_second_commit(remote, capsys):
    # Covers AE6.
    write_export([{"game": "zelda"}])
    publish.main()
    publish.main()
    assert len(remote_log(remote)) == 1
    assert "unchanged" in capsys.readouterr().out


def test_a_new_generated_date_alone_is_not_a_change(remote):
    # Covers AE6: weekly runs within a month see the same monthly data.
    write_export([{"game": "zelda"}], generated="2026-10-25")
    publish.main()
    write_export([{"game": "zelda"}], generated="2026-11-01")
    publish.main()
    assert len(remote_log(remote)) == 1
    assert remote_file(remote)["generated"] == "2026-10-25"


def test_changed_file_adds_one_commit(remote):
    write_export([{"game": "zelda"}])
    publish.main()
    write_export([{"game": "sonic"}], generated="2026-11-01")
    publish.main()
    assert remote_log(remote) == ["Update results 2026-11-01", "Update results 2026-10-25"]
    assert remote_file(remote)["games"] == [{"game": "sonic"}]


def test_custom_branch_and_path(remote):
    paths.settings().write_text(f'[publish]\nrepo = "{remote}"\nbranch = "main"\npath = "phone/out.json"\n',
                                encoding="utf-8")
    write_export()
    publish.main()
    assert remote_file(remote, "phone/out.json")["schema_version"] == 1


def test_unreachable_remote_raises_and_keeps_the_local_export(remote, tmp_path):
    paths.settings().write_text(f'[publish]\nrepo = "{tmp_path / "missing.git"}"\n', encoding="utf-8")
    write_export()
    with pytest.raises(publish.PublishError):
        publish.main()
    assert paths.export().exists()


def test_missing_export_names_the_stage(remote):
    with pytest.raises(publish.PipelineError, match="run `ra export` first"):
        publish.main()


def test_path_outside_the_repository_is_refused(remote):
    paths.settings().write_text(f'[publish]\nrepo = "{remote}"\npath = "../escape.json"\n', encoding="utf-8")
    write_export()
    with pytest.raises(publish.PublishError, match="inside the repository"):
        publish.main()


def test_public_repository_is_refused(tmp_path, monkeypatch):
    # Covers AE5.
    fake_gh(tmp_path, monkeypatch, "echo false")
    with pytest.raises(publish.PublishError, match="alutterb/retro-analytics-data"):
        publish.check_private(TARGET)


def test_private_repository_passes(tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, 'test "$3" = alutterb/retro-analytics-data && echo true')
    publish.check_private(TARGET)


def test_failing_gh_is_refused(tmp_path, monkeypatch):
    fake_gh(tmp_path, monkeypatch, "echo 'not logged in' >&2; exit 1")
    with pytest.raises(publish.PublishError, match="private"):
        publish.check_private(TARGET)


def test_missing_gh_is_refused(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(publish.PublishError, match="gh"):
        publish.check_private(TARGET)


def test_non_github_repository_is_refused_before_any_git_call(monkeypatch):
    monkeypatch.setattr(publish, "_git", lambda *args, **kwargs: pytest.fail("git called"))
    paths.settings().write_text('[publish]\nrepo = "git@gitlab.com:me/data.git"\n', encoding="utf-8")
    write_export()
    with pytest.raises(publish.PublishError, match="GitHub"):
        publish.main()


def test_credentials_in_the_url_are_refused_without_echoing_them(monkeypatch):
    monkeypatch.setattr(publish, "_git", lambda *args, **kwargs: pytest.fail("git called"))
    with pytest.raises(publish.PublishError, match="credentials") as error:
        publish.check_private("https://me:ghp_secret@github.com/alutterb/retro-analytics-data.git")
    assert "ghp_secret" not in str(error.value)


@pytest.mark.parametrize("url", [
    "git@github.com:alutterb/retro-analytics-data.git",
    "git@github.com:alutterb/retro-analytics-data",
    "ssh://git@github.com/alutterb/retro-analytics-data.git",
    "https://github.com/alutterb/retro-analytics-data",
    "https://github.com/alutterb/retro-analytics-data.git",
])
def test_github_url_forms(url):
    assert publish.github_repo(url) == "alutterb/retro-analytics-data"


def test_cli_registers_publish(monkeypatch):
    called = []
    monkeypatch.setattr(cli.publish, "main", lambda: called.append("publish"))
    cli.main(["publish"])
    assert called == ["publish"]
