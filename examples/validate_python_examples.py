#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Python Script Validator

Recursively executes all .py files under a specified directory,
validates their execution status based on console output,
and generates a detailed summary report.

Success Criteria:
- A script is marked as SUCCESSFUL if:
    a) Its combined stdout/stderr contains NONE of the following substrings:
        - "error" (case-insensitive)
        - "mismatch"
        - "traceback"
  OR
    b) At least one line in the output contains:
        - The standalone word "All"
        - Followed (on the same line) by any of:
            • pass, passed, passing
            • success, successful, successfully
       (e.g., "All tests passed", "INFO: All cases successfully validated")

Failure Criteria:
- If neither condition is met, the script is marked as FAILED.
- The specific triggering keyword(s) are recorded for diagnostics.

Note: This script excludes itself from execution to prevent recursion.
"""

import os
import subprocess
import sys
import argparse
import re
from pathlib import Path


def _has_success_exemption(output: str) -> bool:
    """
    Determine if the output contains an exemption line indicating overall success.

    An exemption line must:
      - Contain the word 'All' as a standalone word (not part of another word)
      - On the same line, contain a success indicator:
          'pass', 'passed', 'passing',
          'success', 'successful', or 'successfully'

    Examples of matching lines:
      - "All 10 tests passed."
      - "[INFO] All examples were successful!"
      - "Final result: All cases successfully completed."

    Returns:
        bool: True if such a line exists; False otherwise.
    """
    # Compile pattern once per call (lightweight due to caching in CPython)
    pattern = re.compile(
        r'\bAll\b.*\b(?:pass(?:ed|ing)?|success(?:ful(?:ly)?)?)',
        re.IGNORECASE
    )
    return any(pattern.search(line) for line in output.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute and validate all Python scripts in a directory tree."
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="0",
        help="Value for the DEVICE_ID environment variable (default: '0')."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="Root directory to scan for .py files (default: current directory)."
    )
    args = parser.parse_args()

    # Set environment variable for child processes
    env = os.environ.copy()
    env["TILE_FWK_DEVICE_ID"] = args.device_id

    # Resolve and validate target directory
    target_dir = Path(args.directory).resolve()
    if not target_dir.exists():
        print(f"Error: Specified directory '{target_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Exclude this script from execution
    self_path = Path(__file__).resolve()

    # Discover all .py files recursively
    py_files = sorted(target_dir.rglob("*.py"))
    py_files = [f for f in py_files if f.resolve() != self_path]

    # Convert to relative paths and sort for deterministic order
    relative_paths = [str(f.relative_to(target_dir)) for f in py_files]
    relative_paths.sort()

    if not relative_paths:
        print(f"No .py files found in '{target_dir}' (excluding this script).")
        return

    success_count = 0
    fail_count = 0
    failure_details = []  # List of tuples: (script_path, reason_string)

    print(f"DEVICE_ID set to: {args.device_id}")
    print(f"Scanning directory: {target_dir}")
    print(f"Found {len(relative_paths)} .py file(s). Starting execution...\n")

    for rel_path in relative_paths:
        full_path = target_dir / rel_path
        print(f"Running: {rel_path}")

        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5-minute timeout per script
            )
            output = result.stdout + result.stderr

            # Check for exemption condition (line-based)
            has_exemption = _has_success_exemption(output)

            # Detect presence of failure-indicating substrings (case-insensitive)
            lower_output = output.lower()
            detected_keywords = []
            if "error" in lower_output:
                detected_keywords.append("Error")
            if "mismatch" in lower_output:
                detected_keywords.append("mismatch")
            if "traceback" in lower_output:
                detected_keywords.append("Traceback")

            if has_exemption:
                print(f"✅ Success: {rel_path} (exempted via 'All ... pass/success' pattern)")
                success_count += 1
            elif detected_keywords:
                reason = ", ".join(detected_keywords)
                print(f"❌ Failure: {rel_path} (triggered by: {reason})")
                fail_count += 1
                failure_details.append((rel_path, reason))
            else:
                print(f"✅ Success: {rel_path}")
                success_count += 1

        except subprocess.TimeoutExpired:
            msg = "Timeout (exceeded 300s)"
            print(f"❌ Failure: {rel_path} ({msg})")
            fail_count += 1
            failure_details.append((rel_path, msg))
        except Exception as e:
            msg = f"Exception: {e}"
            print(f"❌ Failure: {rel_path} ({msg})")
            fail_count += 1
            failure_details.append((rel_path, msg))

        print("-" * 50)

    # Final summary report
    total = success_count + fail_count
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    print(f"Target directory : {target_dir}")
    print(f"DEVICE_ID        : {args.device_id}")
    print(f"Total scripts    : {total}")
    print(f"✅ Successful     : {success_count}")
    print(f"❌ Failed         : {fail_count}")

    if failure_details:
        print("\nFailed Scripts (with diagnostic reasons):")
        for script, reason in failure_details:
            print(f"  • {script} → {reason}")

    print("=" * 60)


if __name__ == "__main__":
    main()