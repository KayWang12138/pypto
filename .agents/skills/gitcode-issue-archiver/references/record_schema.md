# Archive Record Schema

The archiver creates and maintains a JSON record file to track issue state and enable incremental updates.

## Location

`{archive_dir}/archive_record.json`

## Schema

```json
{
  "last_check": "ISO 8601 timestamp",
  "repository": "owner/repo",
  "max_issue_number": 42,
  "issues": {
    "0": {
      "exists": true,
      "state": "open|closed",
      "title": "Issue title",
      "updated_at": "ISO 8601 timestamp",
      "comment_count": 5
    },
    "1": {
      "exists": false,
      "reason": "deleted|error",
      "error": "Error message",
      "last_checked": "ISO 8601 timestamp"
    }
  }
}
```

## Fields

### Top-Level

- `last_check` (string, required): Last time archiver ran (ISO 8601 format)
- `repository` (string, required): Repository path (`owner/repo`)
- `max_issue_number` (integer, required): Highest issue number discovered
- `issues` (object, required): Map of issue number → issue record

### Issue Record (exists: true)

- `exists` (boolean, required): Always `true` for existing issues
- `state` (string, required): Issue state - `"open"` or `"closed"`
- `title` (string, required): Issue title
- `updated_at` (string, required): Last update timestamp (ISO 8601)
- `comment_count` (integer, optional): Number of comments

### Issue Record (exists: false)

- `exists` (boolean, required): Always `false` for deleted/missing issues
- `reason` (string, required): `"deleted"` or `"error"`
- `error` (string, optional): Error details (only when reason is `"error"`)
- `last_checked` (string, required): When this issue was last checked (ISO 8601)

## Usage

Archiver uses record to determine if issue needs update:

1. **New issue**: Not in record → fetch
2. **Deleted issue**: In record with `exists: true` → mark deleted
3. **State change**: Record state ≠ current state → fetch
4. **Timestamp change**: Record `updated_at` < current `updated_at` → fetch
5. **Unchanged**: Record state and timestamp match → skip

## Example

```json
{
  "last_check": "2026-01-24T10:30:00Z",
  "repository": "cann/pypto",
  "max_issue_number": 42,
  "issues": {
    "0": {
      "exists": true,
      "state": "closed",
      "title": "Initial issue",
      "updated_at": "2026-01-23T15:30:00Z",
      "comment_count": 3
    },
    "1": {
      "exists": false,
      "reason": "deleted",
      "last_checked": "2026-01-24T10:30:00Z"
    },
    "2": {
      "exists": true,
      "state": "open",
      "title": "Feature request",
      "updated_at": "2026-01-24T09:15:00Z",
      "comment_count": 7
    }
  }
}
```
