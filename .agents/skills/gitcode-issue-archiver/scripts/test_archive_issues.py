#!/usr/bin/env python3

import json
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from archive_issues_mcp import GitCodeMCPArchiver


def test_markdown_generation():
    print("Testing markdown generation...")

    issue = {
        'number': 42,
        'title': 'Test Issue',
        'state': 'open',
        'user': {'login': 'testuser'},
        'created_at': '2026-01-24T10:00:00Z',
        'updated_at': '2026-01-24T11:00:00Z',
        'html_url': 'https://gitcode.com/test/repo/issues/42',
        'body': 'This is a test issue description.',
        'labels': [],
        'assignee': None,
        'milestone': None
    }

    comments = [
        {
            'user': {'login': 'commenter1'},
            'created_at': '2026-01-24T10:30:00Z',
            'body': 'First comment'
        }
    ]

    archiver = GitCodeMCPArchiver('test/repo', '/tmp/test')
    markdown = archiver._generate_markdown(issue, comments)

    assert '# Issue #42: Test Issue' in markdown, "Issue title missing"
    assert '**State**: open' in markdown, "State missing"
    assert '**Author**: testuser' in markdown, "Author missing"
    assert 'This is a test issue description.' in markdown, "Body missing"
    assert 'First comment' in markdown, "Comment missing"

    print("  ✓ Markdown generation passed")
    return True


def test_markdown_no_comments():
    print("Testing markdown with zero comments...")

    issue = {
        'number': 1,
        'title': 'No Comments Issue',
        'state': 'closed',
        'user': {'login': 'author'},
        'created_at': '2026-01-24T10:00:00Z',
        'updated_at': '2026-01-24T11:00:00Z',
        'body': 'Body text.',
        'labels': [],
    }

    archiver = GitCodeMCPArchiver('test/repo', '/tmp/test')
    markdown = archiver._generate_markdown(issue, [])

    assert '# Issue #1: No Comments Issue' in markdown
    assert '## Comments' not in markdown, "Should not have Comments section for empty list"

    print("  ✓ Markdown with zero comments passed")
    return True


def test_needs_update():
    print("Testing update detection...")

    archiver = GitCodeMCPArchiver('test/repo', '/tmp/test')

    record = {'issues': {}}
    issue = {'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}
    assert archiver._needs_update(record, 1, issue) is True, "Should update new issue"

    record = {'issues': {'1': {'exists': True, 'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}}}
    issue = {'state': 'closed', 'updated_at': '2026-01-24T10:00:00Z'}
    assert archiver._needs_update(record, 1, issue) is True, "Should update on state change"

    record = {'issues': {'1': {'exists': True, 'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}}}
    issue = {'state': 'open', 'updated_at': '2026-01-24T11:00:00Z'}
    assert archiver._needs_update(record, 1, issue) is True, "Should update on timestamp change"

    record = {'issues': {'1': {'exists': True, 'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}}}
    issue = {'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}
    assert archiver._needs_update(record, 1, issue) is False, "Should not update when unchanged"

    record = {'issues': {'1': {'exists': False, 'reason': 'deleted'}}}
    issue = {'state': 'open', 'updated_at': '2026-01-24T10:00:00Z'}
    assert archiver._needs_update(record, 1, issue) is True, "Should update deleted issue that now exists"

    print("  ✓ Update detection logic passed")
    return True


def test_record_operations():
    print("Testing record file operations...")

    temp_dir = tempfile.mkdtemp()
    try:
        archiver = GitCodeMCPArchiver('test/repo', temp_dir)

        record = archiver._create_empty_record()
        record['issues']['1'] = {'exists': True, 'state': 'open'}
        archiver._save_record(record)

        loaded = archiver._load_record()

        assert 'repository' in loaded, "Repository field missing"
        assert 'issues' in loaded, "Issues field missing"
        assert loaded['issues']['1']['state'] == 'open', "Issue data not preserved"

        print("  ✓ Record file operations passed")
        return True
    finally:
        shutil.rmtree(temp_dir)


def test_markdown_saving():
    print("Testing markdown file saving...")

    temp_dir = tempfile.mkdtemp()
    try:
        archiver = GitCodeMCPArchiver('test/repo', temp_dir)
        archiver._save_issue_markdown(42, '# Test Issue\nTest content')

        file_path = Path(temp_dir) / 'issue-42.md'
        assert file_path.exists(), "Markdown file not created"

        with open(file_path, 'r') as f:
            content = f.read()
            assert '# Test Issue' in content, "Content not saved correctly"

        print("  ✓ Markdown file saving passed")
        return True
    finally:
        shutil.rmtree(temp_dir)


def test_config_operations():
    print("Testing config save/load with repo_path...")

    import archive_issues_mcp as mod
    original_config = mod.CONFIG_FILE

    temp_dir = tempfile.mkdtemp()
    try:
        mod.CONFIG_FILE = Path(temp_dir) / "config.json"

        GitCodeMCPArchiver._save_config("/workspace/archive/test", "owner/repo")
        config = GitCodeMCPArchiver._load_config()

        assert config["last_archive_dir"] == "/workspace/archive/test", "archive_dir not saved"
        assert config["repo_path"] == "owner/repo", "repo_path not saved"

        print("  ✓ Config operations passed")
        return True
    finally:
        mod.CONFIG_FILE = original_config
        shutil.rmtree(temp_dir)


def main():
    print("=" * 50)
    print("GitCode Issue Archiver - Unit Tests")
    print("=" * 50)
    print()

    tests = [
        test_markdown_generation,
        test_markdown_no_comments,
        test_needs_update,
        test_record_operations,
        test_markdown_saving,
        test_config_operations,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
