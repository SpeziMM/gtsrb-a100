"""Check staged files and current-branch history without printing secret values."""
import argparse
from pathlib import PurePosixPath
import re
import subprocess


PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "credential in URL": re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
    "assigned secret": re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|session[_-]?key|password)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+_=.-]{16,}"),
    "JWT session token": re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
}


def private_path(name):
    parts = PurePosixPath(name).parts
    for part in parts:
        lower = part.lower()
        if (lower.startswith(".venv") or lower in ("venv", "env", ".conda", ".ssh", ".aws", ".azure", ".codex", ".config", ".cache", ".git-credentials", ".netrc", ".npmrc", ".pypirc", ".envrc")
                or lower == ".env" or lower.startswith(".env.")
                or lower.startswith(("id_rsa", "id_ed25519"))):
            return True
    suffix = PurePosixPath(name).suffix.lower()
    base = parts[-1].lower()
    return suffix in (".env", ".pem", ".key", ".p12", ".pfx", ".jks", ".token") or (
        base.startswith(("credentials", "secrets", "session")) and suffix == ".json")


def git(*args):
    return subprocess.check_output(["git", *args])


def scan(history=False):
    files = set()
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            _, oid, _ = header.split()
            files.add((oid.decode(), name.decode()))
    if history:
        # Include publishable ancestry, not local reflog/stash backups.
        for commit in git("rev-list", "HEAD").decode().splitlines():
            for entry in git("ls-tree", "-r", "-z", commit).split(b"\0"):
                if entry:
                    header, name = entry.split(b"\t", 1)
                    _, kind, oid = header.split()
                    if kind == b"blob":
                        files.add((oid.decode(), name.decode()))
    findings = set()
    contents = {}
    for oid, name in files:
        if private_path(name):
            findings.add((name, "private/environment path"))
        if oid not in contents:
            contents[oid] = git("cat-file", "blob", oid)
        for label, pattern in PATTERNS.items():
            if pattern.search(contents[oid]):
                findings.add((name, label))
    return sorted(findings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="also inspect current branch ancestry")
    args = parser.parse_args()
    findings = scan(args.history)
    for path, reason in findings:
        print("{}: {}".format(path, reason))
    if findings:
        raise SystemExit("Audit failed; remove these files/secrets from the proposed publication")
    print("Audit passed: no private/environment paths or matching secret patterns in the inspected files")
