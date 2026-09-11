"""
Parse a `python -X importtime` log and print the ancestor chain
(who ultimately caused a module to be imported) for a target module.

python -X importtime prints in POSTORDER: a module's own line is printed
only after all of its dependencies have finished importing. So a line's
parent is the NEXT line (scanning forward) with a strictly smaller indent.

Usage:
    python find_import_parent_v2.py import2.log torch
    python find_import_parent_v2.py import2.log transformers
"""
import re
import sys

LINE_RE = re.compile(r"^import time:\s*(\d+)\s*\|\s*(\d+)\s*\|(\s*)(.+)$")

def read_lines(path):
    """Windows PowerShell '2>' redirection writes UTF-16LE, not UTF-8.
    Try utf-16 first (auto-handles BOM/LE/BE), fall back to utf-8."""
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                lines = f.readlines()
            if any(LINE_RE.match(l.rstrip("\n")) for l in lines[:200]):
                return lines
        except (UnicodeError, UnicodeDecodeError):
            continue
    return []

def parse(path):
    """Handles PowerShell wrapping long lines: when the module name gets
    pushed past terminal width, it lands on the NEXT physical line instead
    of the same one. Detect empty-name matches and pull the name from the
    following raw line in that case."""
    entries = []
    lines = [l.rstrip("\n") for l in read_lines(path)]
    i = 0
    while i < len(lines):
        raw = lines[i]
        m = LINE_RE.match(raw)
        if m:
            self_us, cum_us, indent, name = m.groups()
            name = name.strip()
            if not name and i + 1 < len(lines):
                # name wrapped onto the next line
                nxt = lines[i + 1]
                if not LINE_RE.match(nxt):
                    name = nxt.strip()
                    i += 1  # consume the wrapped continuation line
            if name:
                entries.append([int(self_us), int(cum_us), len(indent), name])
        i += 1
    return entries

def find_parent(entries, index):
    """Scan forward from index for the next shallower-indent line."""
    _, _, depth, _ = entries[index]
    for j in range(index + 1, len(entries)):
        if entries[j][2] < depth:
            return j
    return None

def ancestry_chain(entries, target_name):
    matches = [i for i, e in enumerate(entries) if e[3] == target_name]
    if not matches:
        print(f"'{target_name}' not found in log.")
        return
    for i in matches:
        self_us, cum_us, depth, name = entries[i]
        chain = [name]
        idx = i
        while True:
            p = find_parent(entries, idx)
            if p is None:
                break
            chain.append(entries[p][3])
            idx = p
        chain.reverse()
        print(f"--- match at line-entry {i} (self={self_us}us cum={cum_us}us) ---")
        print("  " + "\n  -> ".join(chain))
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_import_parent_v2.py <logfile> <module_name>")
        sys.exit(1)
    entries = parse(sys.argv[1])
    if not entries:
        print("No entries parsed - check file path/format.")
        sys.exit(1)
    ancestry_chain(entries, sys.argv[2])
