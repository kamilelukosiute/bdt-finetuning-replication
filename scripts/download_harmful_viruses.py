#!/usr/bin/env python3
"""
Download harmful human-infecting virus genomes from NCBI.

Reads the harmful_virus_species.tsv file and downloads all RefSeq
genome sequences using NCBI Entrez/efetch API.

For segmented viruses, concatenates all segments into a single sequence
(separated by 100 N's as spacer, following common convention).

Output: data/harmful_virus_genomes.fasta
"""

import os
import sys
import time
import csv
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_TSV = PROJECT_DIR / "data" / "harmful_virus_species.tsv"
OUTPUT_FASTA = PROJECT_DIR / "data" / "harmful_virus_genomes.fasta"
OUTPUT_CONCAT = PROJECT_DIR / "data" / "harmful_virus_genomes_concat.fasta"

ENTREZ_EMAIL = "lukosiutekamile@gmail.com"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEGMENT_SPACER = "N" * 100  # spacer between concatenated segments


def fetch_fasta(accession: str, retries: int = 3, delay: float = 0.4) -> str:
    """Fetch a FASTA sequence from NCBI by accession number."""
    url = f"{NCBI_BASE}?db=nucleotide&id={accession}&rettype=fasta&retmode=text&email={ENTREZ_EMAIL}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'BDT-replication/1.0')
            with urllib.request.urlopen(req, timeout=30) as response:
                text = response.read().decode('utf-8')
                if text.strip().startswith('>'):
                    return text.strip()
                else:
                    print(f"  WARNING: unexpected response for {accession}: {text[:100]}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Attempt {attempt+1}/{retries} failed for {accession}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    print(f"  ERROR: Failed to fetch {accession} after {retries} attempts")
    return None


def parse_fasta_sequence(fasta_text: str) -> tuple:
    """Parse FASTA text into (header, sequence)."""
    lines = fasta_text.strip().split('\n')
    header = lines[0]
    sequence = ''.join(line.strip() for line in lines[1:] if not line.startswith('>'))
    return header, sequence


def main():
    if not INPUT_TSV.exists():
        print(f"ERROR: Input file not found: {INPUT_TSV}")
        sys.exit(1)

    # Parse the TSV file
    viruses = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row['virus_name'].startswith('#'):
                continue
            viruses.append(row)

    print(f"Loaded {len(viruses)} virus entries from {INPUT_TSV}")

    # Collect all unique accessions
    all_accessions = set()
    for v in viruses:
        for acc in v['accessions'].split(','):
            all_accessions.add(acc.strip())

    print(f"Total unique accessions to fetch: {len(all_accessions)}")

    # Download all accessions
    fasta_cache = {}
    for i, acc in enumerate(sorted(all_accessions)):
        print(f"[{i+1}/{len(all_accessions)}] Fetching {acc}...")
        fasta_text = fetch_fasta(acc)
        if fasta_text:
            fasta_cache[acc] = fasta_text
        time.sleep(0.35)  # NCBI rate limit: ~3 requests/sec without API key

    print(f"\nSuccessfully fetched {len(fasta_cache)}/{len(all_accessions)} accessions")

    # Write individual segment FASTA (all segments as separate entries)
    with open(OUTPUT_FASTA, 'w') as f:
        for v in viruses:
            accessions = [a.strip() for a in v['accessions'].split(',')]
            for acc in accessions:
                if acc in fasta_cache:
                    f.write(fasta_cache[acc] + '\n')
                else:
                    print(f"WARNING: Missing {acc} for {v['virus_name']}")

    print(f"Wrote individual segments to {OUTPUT_FASTA}")

    # Write concatenated FASTA (segments joined per virus)
    with open(OUTPUT_CONCAT, 'w') as f:
        for v in viruses:
            accessions = [a.strip() for a in v['accessions'].split(',')]
            sequences = []
            missing = False
            for acc in accessions:
                if acc in fasta_cache:
                    _, seq = parse_fasta_sequence(fasta_cache[acc])
                    sequences.append(seq)
                else:
                    missing = True
                    print(f"WARNING: Missing {acc} for {v['virus_name']}, skipping concat")

            if not missing and sequences:
                if len(sequences) == 1:
                    concat_seq = sequences[0]
                else:
                    concat_seq = SEGMENT_SPACER.join(sequences)

                # Clean virus name for FASTA header
                clean_name = v['virus_name'].replace(' ', '_')
                category = v['category']
                total_len = len(concat_seq.replace('N', '')) if SEGMENT_SPACER in concat_seq else len(concat_seq)

                header = f">{clean_name} category={category} accessions={v['accessions']} length={len(concat_seq)} coding_length={total_len}"

                # Write sequence in 80-char lines
                f.write(header + '\n')
                for j in range(0, len(concat_seq), 80):
                    f.write(concat_seq[j:j+80] + '\n')

    print(f"Wrote concatenated genomes to {OUTPUT_CONCAT}")

    # Summary statistics
    print("\n=== Summary by category ===")
    from collections import Counter
    cat_counts = Counter(v['category'] for v in viruses)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} viruses")
    print(f"  TOTAL: {sum(cat_counts.values())} viruses")

    # Check for any failures
    failed = all_accessions - set(fasta_cache.keys())
    if failed:
        print(f"\n!!! {len(failed)} accessions failed to download:")
        for acc in sorted(failed):
            print(f"  {acc}")


if __name__ == "__main__":
    main()
