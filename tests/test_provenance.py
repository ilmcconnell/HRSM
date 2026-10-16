import logging
import subprocess

from hrsmpipe.provenance import provenance_table


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo(path):
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("a")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "one")


def test_clean_tagged_checkout(tmp_path):
    _repo(tmp_path)
    _git(tmp_path, "tag", "v1.0")
    p = provenance_table(tmp_path)
    assert p["git_tag"] == "v1.0" and p["git_commits_since_tag"] == 0
    assert p["git_dirty"] is False
    assert len(p["git_commit"]) == 40
    assert p["git_describe"].startswith("v1.0-0-g")
    assert p["python"] and p["created_utc"].endswith("Z") and p["hrsmpipe_version"]


def test_commits_since_tag_and_dirty(tmp_path):
    _repo(tmp_path)
    _git(tmp_path, "tag", "v1.0")
    (tmp_path / "g.txt").write_text("b")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "two")
    (tmp_path / "f.txt").write_text("changed")
    p = provenance_table(tmp_path)
    assert p["git_commits_since_tag"] == 1 and p["git_dirty"] is True
    assert p["git_describe"].endswith("-dirty")


def test_untagged_repo_has_null_tag(tmp_path):
    _repo(tmp_path)
    p = provenance_table(tmp_path)
    assert p["git_tag"] is None and p["git_commits_since_tag"] is None
    assert p["git_commit"]


def test_non_git_directory(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        p = provenance_table(tmp_path)
    assert p["git_describe"] is None and p["git_commit"] is None and p["git_dirty"] is None
    assert "git_* provenance" in caplog.text
