import argparse
import os
import re


def _clean_wikitext(text: str, *, drop_headings: bool) -> str:
    out_lines: list[str] = []

    heading_re = re.compile(r"^=+\s*.*?\s*=+$")

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            continue

        if drop_headings and heading_re.fullmatch(line):
            continue

        # WikiText "escaped punctuation"
        line = line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")

        # Join hyphenated words ("role - playing" -> "role-playing")
        line = re.sub(r"\s*-\s*", "-", line)

        # Fix spacing around punctuation and brackets
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"\s+([)\]])", r"\1", line)
        line = re.sub(r"([(\[])\s+", r"\1", line)

        # Contractions / possessives ("game 's" -> "game's", "don 't" -> "don't")
        line = re.sub(r"\b(\w+)\s+'(s|re|ve|d|ll|m|t)\b", r"\1'\2", line)

        # Remove padding inside quotes.
        line = re.sub(r"\"\s+", "\"", line)  # after opening quote
        line = re.sub(r"\s+\"(?=[,.;:!?])", "\"", line)  # before closing quote
        line = re.sub(r"\s+\"$", "\"", line)  # before closing quote at EOL

        # Thousands separators ("102, 779" -> "102,779"), but keep date commas ("3, 1986").
        line = re.sub(r"(\d),\s+(\d{3})\b", r"\1,\2", line)
        # Date commas ("April 3,1986" -> "April 3, 1986").
        line = re.sub(r"(\d),(?=\d{4}\b)", r"\1, ", line)

        # Collapse whitespace
        line = re.sub(r"\s+", " ", line).strip()
        out_lines.append(line)

    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "\n".join(out_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a WikiText-style raw .txt file.")
    parser.add_argument("--input", required=True, help="Input raw text file (UTF-8).")
    parser.add_argument("--output", required=True, help="Output cleaned text file (UTF-8).")
    parser.add_argument(
        "--keep_headings",
        action="store_true",
        help="Keep lines like '== Heading =='.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = _clean_wikitext(raw, drop_headings=not args.keep_headings)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(
        f"wrote {args.output} | chars {len(raw)} -> {len(cleaned)} | lines {raw.count(chr(10))+1} -> {cleaned.count(chr(10))}"
    )


if __name__ == "__main__":
    main()
