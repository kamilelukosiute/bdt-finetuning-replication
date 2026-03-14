#!/usr/bin/env python3
"""Generate the segment-level TSV from the species-level TSV."""
import csv
from pathlib import Path

PROJECT = Path(__file__).parent.parent
species_file = PROJECT / "data" / "harmful_virus_species.tsv"
segment_file = PROJECT / "data" / "harmful_virus_genomes.tsv"

with open(species_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    species = [r for r in reader if not r['virus_name'].startswith('#')]

print(f"Read {len(species)} species")
print(f"Writing segment-level TSV to {segment_file}")

with open(segment_file, 'w') as f:
    f.write("virus_name\tcategory\taccession\tnum_segments\tfamily\tnotes\n")
    for s in species:
        accessions = [a.strip() for a in s['accessions'].split(',')]
        for i, acc in enumerate(accessions):
            if len(accessions) == 1:
                name = s['virus_name']
            else:
                name = f"{s['virus_name']} segment {i+1}"
            f.write(f"{name}\t{s['category']}\t{acc}\t{s['num_segments']}\t{s['family']}\t{s['notes']}\n")

print("Done")
