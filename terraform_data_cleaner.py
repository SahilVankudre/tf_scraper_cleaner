import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_FILE_CHARS = 50
MAX_FILE_CHARS = 50_000

# Sensitive data patterns to detect and sanitize
SENSITIVE_PATTERNS = [
    (r'(?i)(password\s*=\s*")[^"]+(")',        'var.password'),
    (r'(?i)(secret\s*=\s*")[^"]+(")',          'var.secret'),
    (r'(?i)(api[_-]?key\s*=\s*")[^"]+(")',     'var.api_key'),
    (r'(?i)(access[_-]?key\s*=\s*")[^"]+(")', 'var.access_key'),
    (r'(?i)(private[_-]?key\s*=\s*")[^"]+(")', 'var.private_key'),
    (r'(?i)(token\s*=\s*")[^"]+(")',           'var.token'),
    (r'AKIA[0-9A-Z]{16}',                      'var.aws_access_key'),
    (r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END[^-]+-----', '# PRIVATE KEY REMOVED'),
]


# ─────────────────────────────────────────────
# CLEANING FUNCTIONS
# ─────────────────────────────────────────────
def compute_hash(content: str) -> str:
    """Hash normalized content for deduplication."""
    normalized = re.sub(r'#.*$',      '', content, flags=re.MULTILINE)
    normalized = re.sub(r'//.*$',     '', normalized, flags=re.MULTILINE)
    normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
    normalized = re.sub(r'\s+',       ' ', normalized).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def is_valid_terraform(content: str) -> bool:
    """Basic Terraform syntax check."""
    has_block = bool(re.search(
        r'\b(resource|module|data|variable|output|locals|terraform|provider)\s+"',
        content
    ))
    balanced  = content.count('{') == content.count('}') and content.count('{') > 0
    non_comment_lines = [
        l for l in content.splitlines()
        if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')
    ]
    return has_block and balanced and len(non_comment_lines) > 2


def sanitize_sensitive(content: str) -> tuple[str, List[str]]:
    """Replace sensitive values with variable references. Returns (cleaned, found_list)."""
    found = []
    for pattern, replacement in SENSITIVE_PATTERNS:
        if re.search(pattern, content):
            found.append(pattern)
            # Preserve the key name, replace only the value
            if '(' in pattern:
                content = re.sub(pattern, lambda m: f'{m.group(1)}{replacement}{m.group(2)}', content)
            else:
                content = re.sub(pattern, replacement, content)
    return content, found


def remove_comments(content: str) -> str:
    """
    Remove single-line and multi-line comments.
    Preserves doc-style comments (## or ///).
    Skips comment chars inside strings.
    """
    # Remove block comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.lstrip()

        # Preserve doc comments
        if stripped.startswith('##') or stripped.startswith('///'):
            cleaned_lines.append(line)
            continue

        # Skip pure comment lines
        if stripped.startswith('#') or stripped.startswith('//'):
            continue

        # Strip inline comments (not inside strings)
        in_str, result, i = False, [], 0
        while i < len(line):
            ch = line[i]
            if ch in ('"', "'"):
                in_str = not in_str
                result.append(ch)
            elif not in_str and (ch == '#' or line[i:i+2] == '//'):
                break
            else:
                result.append(ch)
            i += 1

        stripped_result = ''.join(result).rstrip()
        if stripped_result:
            cleaned_lines.append(stripped_result)

    return '\n'.join(cleaned_lines)


def standardize_formatting(content: str) -> str:
    """Normalize whitespace, indentation, blank lines."""
    # Tabs → 2 spaces
    content = content.replace('\t', '  ')

    # Remove trailing whitespace per line
    lines = [l.rstrip() for l in content.splitlines()]

    # Collapse multiple blank lines into one
    cleaned, prev_blank = [], False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = blank

    # Strip leading/trailing blank lines
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return '\n'.join(cleaned)


def clean_file(content: str, seen_hashes: set) -> tuple[Optional[str], str]:
    """
    Full cleaning pipeline for a single .tf file.
    Returns (cleaned_content | None, reason_if_skipped)
    """
    # Size check (raw)
    if len(content) < MIN_FILE_CHARS:
        return None, "too_small"
    if len(content) > MAX_FILE_CHARS:
        return None, "too_large"

    # Deduplication
    if compute_hash(content) in seen_hashes:
        return None, "duplicate"
    seen_hashes.add(compute_hash(content))

    # Syntax validation
    if not is_valid_terraform(content):
        return None, "invalid_syntax"

    # Sensitive data sanitization
    content, sensitive_found = sanitize_sensitive(content)

    # Remove comments
    content = remove_comments(content)

    # Standardize formatting
    content = standardize_formatting(content)

    # Size check again after cleaning
    if len(content) < MIN_FILE_CHARS:
        return None, "too_small_after_clean"

    return content, "ok"


# ─────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────
def build_training_sample(provider: str, service: str,
                          module: str, filename: str, content: str) -> Dict:
    """
    Build a single training sample in instruction-response format
    suitable for fine-tuning (compatible with most fine-tuning frameworks).
    """
    # Derive a natural language description from metadata
    cloud_map = {
        "aws": "AWS", "google": "GCP",
        "azurerm": "Azure", "oci": "Oracle Cloud"
    }
    cloud     = cloud_map.get(provider, provider.upper())
    folder    = filename.split("__")[0] if "__" in filename else "examples"
    tf_file   = filename.split("__")[-1] if "__" in filename else filename

    instruction = (
        f"Write Terraform code for {cloud} {service.upper()} "
        f"({tf_file.replace('.tf','').replace('_',' ')} configuration)."
    )

    return {
        "instruction": instruction,
        "input":       "",
        "output":      content,
        "metadata": {
            "provider": provider,
            "service":  service,
            "module":   module,
            "file":     filename,
            "folder":   folder
        }
    }


# ─────────────────────────────────────────────
# MAIN CLEANER
# ─────────────────────────────────────────────
class TerraformDataCleaner:

    def __init__(self, input_dir: str = "terraform_training_data",
                 output_dir: str = "terraform_cleaned_data"):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.seen_hashes = set()
        self.stats = {
            "total":              0,
            "cleaned":            0,
            "duplicate":          0,
            "too_small":          0,
            "too_large":          0,
            "invalid_syntax":     0,
            "too_small_after_clean": 0,
            "sensitive_sanitized": 0
        }

    def process_provider(self, provider_dir: Path) -> List[Dict]:
        """Process all .tf files under a provider directory."""
        provider = provider_dir.name
        samples  = []

        print(f"\n{'='*60}")
        print(f"  Provider: {provider.upper()}")
        print(f"{'='*60}")

        # Structure: provider/service/module/file.tf
        for service_dir in sorted(provider_dir.iterdir()):
            if not service_dir.is_dir():
                continue
            service = service_dir.name
            print(f"\n  [service] {service}")

            for module_dir in sorted(service_dir.iterdir()):
                if not module_dir.is_dir():
                    continue
                module = module_dir.name

                for tf_file in sorted(module_dir.glob("*.tf")):
                    self.stats["total"] += 1
                    content = tf_file.read_text(encoding="utf-8", errors="ignore")

                    cleaned, reason = clean_file(content, self.seen_hashes)

                    if cleaned is None:
                        self.stats[reason] = self.stats.get(reason, 0) + 1
                        continue

                    # Save cleaned .tf file
                    dest = (self.output_dir / provider / service / module / tf_file.name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(cleaned, encoding="utf-8")

                    # Build training sample
                    sample = build_training_sample(
                        provider, service, module, tf_file.name, cleaned
                    )
                    samples.append(sample)
                    self.stats["cleaned"] += 1

        return samples

    def clean_all(self):
        """Run full cleaning pipeline across all providers."""
        print(f"\n{'='*60}")
        print("  TERRAFORM DATA CLEANER")
        print(f"  Input : {self.input_dir.resolve()}")
        print(f"  Output: {self.output_dir.resolve()}")
        print(f"{'='*60}")

        all_samples = []

        for provider_dir in sorted(self.input_dir.iterdir()):
            if not provider_dir.is_dir() or provider_dir.name.startswith('.'):
                continue
            samples = self.process_provider(provider_dir)
            all_samples.extend(samples)

        # ── Save JSONL (for fine-tuning frameworks: LLaMA-Factory, Axolotl, etc.)
        jsonl_path = self.output_dir / "training_dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")

        # ── Save JSON (for inspection)
        json_path = self.output_dir / "training_dataset.json"
        json_path.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")

        # ── Save metadata summary
        summary = {
            "total_training_samples": len(all_samples),
            "cleaning_stats": self.stats,
            "sample_preview": all_samples[:2] if all_samples else []
        }
        summary_path = self.output_dir / "cleaning_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # ── Print stats
        print(f"\n{'='*60}")
        print("  CLEANING STATS")
        print(f"{'='*60}")
        print(f"  Total files processed   : {self.stats['total']}")
        print(f"  Successfully cleaned    : {self.stats['cleaned']}")
        print(f"  Duplicates removed      : {self.stats['duplicate']}")
        print(f"  Invalid syntax removed  : {self.stats['invalid_syntax']}")
        print(f"  Too small               : {self.stats['too_small']}")
        print(f"  Too large               : {self.stats['too_large']}")
        print(f"  Sensitive data sanitized: {self.stats['sensitive_sanitized']}")
        print(f"\n  Training samples ready  : {len(all_samples)}")
        print(f"  JSONL path              : {jsonl_path}")
        print(f"  JSON path               : {json_path}")
        print(f"{'='*60}")
        print("  ✓ Cleaning complete!")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = TerraformDataCleaner(
        input_dir="terraform_training_data",
        output_dir="terraform_cleaned_data"
    )
    cleaner.clean_all()