#!/usr/bin/env python3
"""Check that every formula in a Markdown file will render on GitHub.

GitHub renders maths with KaTeX inside a Markdown pipeline that has already
consumed some backslashes, so a formula that is valid LaTeX can still fail
there.  These are the failure modes this repository has actually hit, each of
which silently produces a red error box instead of a formula:

  \\[ ... \\] and \\( ... \\)   not delimiters on GitHub; use $ and $$.
  \\operatorname               rejected as "macros not allowed"; use \\mathrm.
  \\{  \\}  \\|  \\lvert-less  Markdown eats the backslash before the brace or
                              bar, leaving KaTeX with an unmatched delimiter.
                              Use \\lbrace \\rbrace \\Vert \\lvert \\rvert.
  CJK inside $...$            KaTeX throws on the characters outright.
  $ ... $ spanning a newline  the inline form must stay on one line.
  $$ with no blank line        a display block glued to the paragraph above it is
    before or after it         swallowed into that paragraph and never rendered.
                              This is the failure that actually bit this repo:
                              54 formulas across two reports looked fine in the
                              source and were invisible on GitHub.

Run it on every document before pushing; a formula nobody can read is worse
than no formula, because the reader assumes the error is theirs.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

BAD_MACRO = re.compile(r"\\operatorname")
BAD_DELIM = re.compile(r"\\[\[\]()]")
EATEN = re.compile(r"\\[{}|]")
CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def problems(path: Path) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    in_display = False
    in_code = False
    for number, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if BAD_DELIM.search(line):
            found.append((number, "\\[ \\] or \\( \\) is not a delimiter on GitHub",
                          line.strip()[:70]))
        if line.strip() == "$$":
            in_display = not in_display
            continue
        segments = [line] if in_display else [
            m.group(1) for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", line)]
        for segment in segments:
            if BAD_MACRO.search(segment):
                found.append((number, "\\operatorname is rejected; use \\mathrm",
                              segment[:70]))
            if EATEN.search(segment):
                found.append((number, "Markdown eats \\{ \\} \\|; use \\lbrace \\rbrace "
                              "\\Vert", segment[:70]))
            if CJK.search(segment):
                found.append((number, "CJK inside maths makes KaTeX throw",
                              segment[:70]))
    # A display block glued to the paragraph above it is swallowed into that
    # paragraph and never rendered.  Only the opening delimiter matters; the
    # closing one is preceded by the formula itself by construction.
    opening = True
    for number, line in enumerate(lines, 1):
        if line.strip() != "$$":
            continue
        if opening:
            before = lines[number - 2].strip() if number >= 2 else ""
            if before:
                found.append((number, "display block needs a blank line before it",
                              before[:70]))
        else:
            after = lines[number].strip() if number < len(lines) else ""
            if after:
                found.append((number, "display block needs a blank line after it",
                              after[:70]))
        opening = not opening

    # an inline $...$ that never closes on its own line
    for number, line in enumerate(lines, 1):
        if line.count("$") % 2 == 1 and "$$" not in line and not line.strip().startswith("```"):
            found.append((number, "odd number of $ on one line; inline maths cannot "
                          "span a newline", line.strip()[:70]))
    return found


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(Path("docs").glob("*.md"))
    total = 0
    for path in targets:
        issues = problems(path)
        total += len(issues)
        status = "OK" if not issues else f"{len(issues)} problem(s)"
        print(f"{path}: {status}")
        for number, why, text in issues[:12]:
            print(f"    line {number}: {why}\n        {text}")
        if len(issues) > 12:
            print(f"    ... and {len(issues) - 12} more")
    print(f"\ntotal: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
