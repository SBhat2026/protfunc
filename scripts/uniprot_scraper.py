#!/usr/bin/env python3
"""
UniProt Metazoa Protein Scraper for ProtFunc Training Data

This script scrapes protein sequences and GO annotations from UniProt,
targeting all Metazoa (taxon 33208) for broad generalization across animal proteins.

Features:
- Async HTTP requests for high throughput
- Resume capability with checkpoint files
- Rate limiting to respect UniProt's guidelines
- Efficient streaming for large datasets
- GO term extraction (MF, BP, CC)
- Taxonomy information for stratified sampling

Usage:
    python uniprot_scraper.py --output data/metazoa_proteins.jsonl --max-proteins 100000
    python uniprot_scraper.py --resume --checkpoint data/checkpoint.json

UniProt REST API docs: https://www.uniprot.org/help/api_queries
"""

import argparse
import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set, AsyncIterator
from urllib.parse import urlencode, quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('uniprot_scraper.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ScraperConfig:
    """Configuration for the UniProt scraper."""
    
    # API settings
    base_url: str = "https://rest.uniprot.org/uniprotkb"
    batch_size: int = 500  # UniProt recommends max 500 per request
    max_retries: int = 5
    retry_delay: float = 2.0
    request_timeout: int = 120
    
    # Rate limiting (UniProt allows ~100 requests/sec for programmatic access)
    requests_per_second: float = 10.0  # Conservative rate
    concurrent_requests: int = 5
    
    # Taxonomy filters
    # Metazoa (33208) is the root for all animals
    # Can add specific taxa for targeted scraping
    taxa: List[int] = field(default_factory=lambda: [
        33208,   # Metazoa (all animals) - default
    ])
    
    # Specific sub-taxa for balanced representation
    sub_taxa: Dict[str, int] = field(default_factory=lambda: {
        'insects': 50557,       # Insecta
        'mammals': 40674,       # Mammalia
        'birds': 8782,          # Aves
        'fish': 7898,           # Actinopterygii (ray-finned fish)
        'amphibians': 8292,     # Amphibia
        'reptiles': 8504,       # Reptilia
        'nematodes': 6231,      # Nematoda
        'mollusks': 6447,       # Mollusca
        'crustaceans': 6657,    # Crustacea
        'arachnids': 6854,      # Arachnida
    })
    
    # GO namespace filters
    go_namespaces: List[str] = field(default_factory=lambda: [
        'molecular_function',
        'biological_process', 
        'cellular_component'
    ])
    
    # Quality filters
    min_sequence_length: int = 30
    max_sequence_length: int = 5000
    reviewed_only: bool = False  # True = Swiss-Prot only, False = include TrEMBL
    with_go_annotations: bool = True  # Only proteins with GO terms
    
    # Output settings
    output_dir: str = "data"
    checkpoint_interval: int = 1000  # Save checkpoint every N proteins


@dataclass 
class ProteinRecord:
    """A single protein record with sequence and annotations."""
    
    accession: str
    entry_name: str
    protein_name: str
    organism: str
    organism_id: int
    sequence: str
    sequence_length: int
    go_terms: List[Dict[str, str]]  # [{id, name, namespace, evidence}]
    taxonomy_lineage: List[str]
    reviewed: bool
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_uniprot_json(cls, entry: dict) -> 'ProteinRecord':
        """Parse a UniProt JSON entry into a ProteinRecord."""
        
        # Extract basic info
        accession = entry.get('primaryAccession', '')
        entry_name = entry.get('uniProtkbId', '')
        
        # Protein name (can be complex structure)
        protein_name = ''
        if 'proteinDescription' in entry:
            pd = entry['proteinDescription']
            if 'recommendedName' in pd:
                protein_name = pd['recommendedName'].get('fullName', {}).get('value', '')
            elif 'submissionNames' in pd and pd['submissionNames']:
                protein_name = pd['submissionNames'][0].get('fullName', {}).get('value', '')
        
        # Organism info
        organism = entry.get('organism', {}).get('scientificName', '')
        organism_id = entry.get('organism', {}).get('taxonId', 0)
        
        # Sequence
        sequence = entry.get('sequence', {}).get('value', '')
        sequence_length = entry.get('sequence', {}).get('length', len(sequence))
        
        # GO terms
        go_terms = []
        for xref in entry.get('uniProtKBCrossReferences', []):
            if xref.get('database') == 'GO':
                go_id = xref.get('id', '')
                props = {p['key']: p['value'] for p in xref.get('properties', [])}
                
                # Parse GO term details
                go_name = props.get('GoTerm', '').split(':')[-1].strip() if 'GoTerm' in props else ''
                namespace_code = go_id.split(':')[0] if ':' in go_id else ''
                
                # Determine namespace from GoTerm prefix (C:, F:, P:)
                namespace = 'unknown'
                if 'GoTerm' in props:
                    prefix = props['GoTerm'][0] if props['GoTerm'] else ''
                    namespace_map = {'F': 'molecular_function', 'P': 'biological_process', 'C': 'cellular_component'}
                    namespace = namespace_map.get(prefix, 'unknown')
                
                go_terms.append({
                    'id': go_id,
                    'name': go_name,
                    'namespace': namespace,
                    'evidence': props.get('GoEvidenceType', '')
                })
        
        # Taxonomy lineage
        taxonomy_lineage = []
        for lineage_item in entry.get('organism', {}).get('lineage', []):
            taxonomy_lineage.append(lineage_item)
        
        # Review status
        reviewed = entry.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'
        
        # Dates
        dates = entry.get('entryAudit', {})
        created_date = dates.get('firstPublicDate')
        modified_date = dates.get('lastSequenceUpdateDate')
        
        return cls(
            accession=accession,
            entry_name=entry_name,
            protein_name=protein_name,
            organism=organism,
            organism_id=organism_id,
            sequence=sequence,
            sequence_length=sequence_length,
            go_terms=go_terms,
            taxonomy_lineage=taxonomy_lineage,
            reviewed=reviewed,
            created_date=created_date,
            modified_date=modified_date
        )


# ============================================================================
# UniProt API Client
# ============================================================================

class UniProtClient:
    """Async client for UniProt REST API."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(config.concurrent_requests)
        self.last_request_time = 0.0
        self.request_interval = 1.0 / config.requests_per_second
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            headers={
                'Accept': 'application/json',
                'User-Agent': 'ProtFunc-Scraper/1.0 (protein function prediction research)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self.rate_limiter:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.request_interval:
                await asyncio.sleep(self.request_interval - elapsed)
            self.last_request_time = time.time()
    
    def _build_query(self, taxon_id: int, cursor: Optional[str] = None) -> str:
        """Build UniProt search query URL."""
        
        # Build query string
        query_parts = [f'(taxonomy_id:{taxon_id})']
        
        if self.config.with_go_annotations:
            query_parts.append('(go:*)')  # Has any GO annotation
        
        if self.config.reviewed_only:
            query_parts.append('(reviewed:true)')
        
        query_parts.append(f'(length:[{self.config.min_sequence_length} TO {self.config.max_sequence_length}])')
        
        query = ' AND '.join(query_parts)
        
        # Build URL parameters
        params = {
            'query': query,
            'format': 'json',
            'size': self.config.batch_size,
            'fields': 'accession,id,protein_name,organism_name,organism_id,sequence,go,lineage,reviewed'
        }
        
        if cursor:
            params['cursor'] = cursor
        
        return f"{self.config.base_url}/search?{urlencode(params, safe=':*[]')}"
    
    async def _fetch_page(self, url: str) -> tuple[List[dict], Optional[str]]:
        """Fetch a single page of results, returning entries and next cursor."""
        
        await self._rate_limit()
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract next cursor from Link header
                    next_cursor = None
                    link_header = response.headers.get('Link', '')
                    if 'rel="next"' in link_header:
                        # Parse cursor from link header
                        for part in link_header.split(','):
                            if 'rel="next"' in part:
                                url_part = part.split(';')[0].strip('<> ')
                                if 'cursor=' in url_part:
                                    next_cursor = url_part.split('cursor=')[1].split('&')[0]
                                break
                    
                    return data.get('results', []), next_cursor
                    
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
        
        return [], None
    
    async def search_taxon(self, taxon_id: int, max_results: Optional[int] = None) -> AsyncIterator[ProteinRecord]:
        """
        Search UniProt for proteins in a given taxon.
        
        Args:
            taxon_id: NCBI taxonomy ID
            max_results: Maximum number of results to return (None for all)
            
        Yields:
            ProteinRecord objects
        """
        url = self._build_query(taxon_id)
        total_fetched = 0
        
        while url:
            entries, next_cursor = await self._fetch_page(url)
            
            for entry in entries:
                try:
                    record = ProteinRecord.from_uniprot_json(entry)
                    
                    # Apply quality filters
                    if record.sequence_length < self.config.min_sequence_length:
                        continue
                    if record.sequence_length > self.config.max_sequence_length:
                        continue
                    if self.config.with_go_annotations and not record.go_terms:
                        continue
                    
                    yield record
                    total_fetched += 1
                    
                    if max_results and total_fetched >= max_results:
                        return
                        
                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue
            
            if next_cursor:
                url = self._build_query(taxon_id, next_cursor)
            else:
                url = None
            
            logger.info(f"Fetched {total_fetched} proteins from taxon {taxon_id}")


# ============================================================================
# Checkpoint Management
# ============================================================================

@dataclass
class Checkpoint:
    """Checkpoint for resuming interrupted scraping."""
    
    completed_taxa: List[int]
    current_taxon: Optional[int]
    current_cursor: Optional[str]
    total_proteins: int
    last_updated: str
    
    def save(self, path: str):
        self.last_updated = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Checkpoint saved: {self.total_proteins} proteins")
    
    @classmethod
    def load(cls, path: str) -> 'Checkpoint':
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def new(cls) -> 'Checkpoint':
        return cls(
            completed_taxa=[],
            current_taxon=None,
            current_cursor=None,
            total_proteins=0,
            last_updated=datetime.now().isoformat()
        )


# ============================================================================
# Main Scraper
# ============================================================================

class MetazoaScraper:
    """Main scraper orchestrating the data collection."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.output_dir / 'checkpoint.json'
        self.output_path = self.output_dir / 'metazoa_proteins.jsonl'
        
        # Statistics
        self.stats = {
            'total': 0,
            'by_taxon': {},
            'by_namespace': {'molecular_function': 0, 'biological_process': 0, 'cellular_component': 0},
            'reviewed': 0,
            'unreviewed': 0
        }
    
    async def run(
        self, 
        max_proteins: Optional[int] = None,
        proteins_per_taxon: Optional[int] = None,
        resume: bool = False,
        taxa_to_scrape: Optional[List[int]] = None
    ):
        """
        Run the scraper.
        
        Args:
            max_proteins: Maximum total proteins to collect
            proteins_per_taxon: Maximum proteins per taxon (for balanced dataset)
            resume: Whether to resume from checkpoint
            taxa_to_scrape: Specific taxa to scrape (default: all sub_taxa)
        """
        # Load or create checkpoint
        if resume and self.checkpoint_path.exists():
            checkpoint = Checkpoint.load(str(self.checkpoint_path))
            logger.info(f"Resuming from checkpoint: {checkpoint.total_proteins} proteins")
        else:
            checkpoint = Checkpoint.new()
        
        # Determine which taxa to scrape
        if taxa_to_scrape:
            taxa = taxa_to_scrape
        else:
            # Use sub-taxa for balanced representation
            taxa = list(self.config.sub_taxa.values())
        
        # Filter out completed taxa
        remaining_taxa = [t for t in taxa if t not in checkpoint.completed_taxa]
        
        logger.info(f"Scraping {len(remaining_taxa)} taxa, {len(checkpoint.completed_taxa)} already complete")
        
        # Open output file in append mode
        mode = 'a' if resume and self.output_path.exists() else 'w'
        
        async with UniProtClient(self.config) as client:
            with open(self.output_path, mode) as outfile:
                for taxon_id in remaining_taxa:
                    taxon_name = self._get_taxon_name(taxon_id)
                    logger.info(f"Starting taxon: {taxon_name} ({taxon_id})")
                    
                    checkpoint.current_taxon = taxon_id
                    taxon_count = 0
                    
                    async for record in client.search_taxon(taxon_id, proteins_per_taxon):
                        # Write record
                        outfile.write(json.dumps(record.to_dict()) + '\n')
                        
                        # Update stats
                        self.stats['total'] += 1
                        taxon_count += 1
                        checkpoint.total_proteins += 1
                        
                        if record.reviewed:
                            self.stats['reviewed'] += 1
                        else:
                            self.stats['unreviewed'] += 1
                        
                        for go_term in record.go_terms:
                            ns = go_term.get('namespace', 'unknown')
                            if ns in self.stats['by_namespace']:
                                self.stats['by_namespace'][ns] += 1
                        
                        # Checkpoint periodically
                        if checkpoint.total_proteins % self.config.checkpoint_interval == 0:
                            checkpoint.save(str(self.checkpoint_path))
                            outfile.flush()
                        
                        # Check total limit
                        if max_proteins and checkpoint.total_proteins >= max_proteins:
                            logger.info(f"Reached max proteins limit: {max_proteins}")
                            checkpoint.save(str(self.checkpoint_path))
                            self._save_stats()
                            return
                    
                    # Mark taxon complete
                    self.stats['by_taxon'][taxon_name] = taxon_count
                    checkpoint.completed_taxa.append(taxon_id)
                    checkpoint.current_taxon = None
                    checkpoint.save(str(self.checkpoint_path))
                    
                    logger.info(f"Completed taxon {taxon_name}: {taxon_count} proteins")
        
        # Final save
        checkpoint.save(str(self.checkpoint_path))
        self._save_stats()
        logger.info(f"Scraping complete: {checkpoint.total_proteins} total proteins")
    
    def _get_taxon_name(self, taxon_id: int) -> str:
        """Get human-readable taxon name."""
        for name, tid in self.config.sub_taxa.items():
            if tid == taxon_id:
                return name
        return str(taxon_id)
    
    def _save_stats(self):
        """Save collection statistics."""
        stats_path = self.output_dir / 'scraping_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")


# ============================================================================
# Data Processing Utilities
# ============================================================================

def prepare_training_data(
    input_path: str,
    output_dir: str,
    test_split: float = 0.1,
    val_split: float = 0.1,
    namespace: str = 'molecular_function',
    min_go_count: int = 10
):
    """
    Process scraped data into training format.
    
    Args:
        input_path: Path to scraped JSONL file
        output_dir: Output directory for processed files
        test_split: Fraction for test set
        val_split: Fraction for validation set
        namespace: GO namespace to filter ('molecular_function', 'biological_process', 'cellular_component')
        min_go_count: Minimum occurrences for a GO term to be included
    """
    import random
    from collections import Counter
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {input_path} for {namespace}")
    
    # First pass: count GO terms
    go_counts = Counter()
    total_proteins = 0
    
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            total_proteins += 1
            for go_term in record.get('go_terms', []):
                if go_term.get('namespace') == namespace:
                    go_counts[go_term['id']] += 1
    
    # Filter GO terms by count
    valid_go_terms = {go_id for go_id, count in go_counts.items() if count >= min_go_count}
    logger.info(f"Found {len(valid_go_terms)} GO terms with >= {min_go_count} occurrences")
    
    # Create GO term to index mapping
    go_to_idx = {go_id: idx for idx, go_id in enumerate(sorted(valid_go_terms))}
    
    # Second pass: process proteins
    proteins = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            
            # Filter GO terms for this namespace
            go_ids = [
                go_term['id'] 
                for go_term in record.get('go_terms', [])
                if go_term.get('namespace') == namespace and go_term['id'] in valid_go_terms
            ]
            
            if not go_ids:
                continue
            
            proteins.append({
                'accession': record['accession'],
                'name': record.get('entry_name', record['accession']),
                'sequence': record['sequence'],
                'organism': record.get('organism', ''),
                'organism_id': record.get('organism_id', 0),
                'go_ids': go_ids,
                'taxonomy': record.get('taxonomy_lineage', [])
            })
    
    logger.info(f"Processed {len(proteins)} proteins with valid GO terms")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(proteins)
    
    n_test = int(len(proteins) * test_split)
    n_val = int(len(proteins) * val_split)
    
    test_data = proteins[:n_test]
    val_data = proteins[n_test:n_test + n_val]
    train_data = proteins[n_test + n_val:]
    
    # Save splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_path = output_dir / f'{split_name}.jsonl'
        with open(split_path, 'w') as f:
            for record in split_data:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Saved {len(split_data)} proteins to {split_path}")
    
    # Save GO term mapping
    go_mapping_path = output_dir / 'go_terms.json'
    with open(go_mapping_path, 'w') as f:
        json.dump({
            'go_to_idx': go_to_idx,
            'idx_to_go': {v: k for k, v in go_to_idx.items()},
            'num_labels': len(go_to_idx),
            'namespace': namespace,
            'go_counts': {k: v for k, v in go_counts.items() if k in valid_go_terms}
        }, f, indent=2)
    logger.info(f"Saved GO term mapping to {go_mapping_path}")
    
    # Save summary statistics
    stats = {
        'total_proteins': len(proteins),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'num_go_terms': len(valid_go_terms),
        'namespace': namespace,
        'min_go_count': min_go_count
    }
    
    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset statistics: {stats}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scrape protein sequences and GO annotations from UniProt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 100K proteins from all Metazoa sub-taxa
  python uniprot_scraper.py --max-proteins 100000
  
  # Scrape 10K proteins per taxon for balanced dataset
  python uniprot_scraper.py --per-taxon 10000
  
  # Resume interrupted scraping
  python uniprot_scraper.py --resume
  
  # Scrape only reviewed (Swiss-Prot) entries
  python uniprot_scraper.py --reviewed-only --max-proteins 50000
  
  # Process scraped data for training
  python uniprot_scraper.py --process data/metazoa_proteins.jsonl --output data/processed
        """
    )
    
    parser.add_argument('--output', '-o', default='data', help='Output directory')
    parser.add_argument('--max-proteins', type=int, help='Maximum total proteins to collect')
    parser.add_argument('--per-taxon', type=int, help='Maximum proteins per taxon')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--reviewed-only', action='store_true', help='Only Swiss-Prot entries')
    parser.add_argument('--taxa', nargs='+', type=int, help='Specific taxon IDs to scrape')
    
    # Processing mode
    parser.add_argument('--process', metavar='FILE', help='Process scraped file into training data')
    parser.add_argument('--namespace', default='molecular_function', 
                        choices=['molecular_function', 'biological_process', 'cellular_component'],
                        help='GO namespace for processing')
    parser.add_argument('--min-go-count', type=int, default=10,
                        help='Minimum GO term occurrences for inclusion')
    
    args = parser.parse_args()
    
    if args.process:
        # Processing mode
        prepare_training_data(
            input_path=args.process,
            output_dir=args.output,
            namespace=args.namespace,
            min_go_count=args.min_go_count
        )
    else:
        # Scraping mode
        config = ScraperConfig(
            output_dir=args.output,
            reviewed_only=args.reviewed_only
        )
        
        scraper = MetazoaScraper(config)
        
        asyncio.run(scraper.run(
            max_proteins=args.max_proteins,
            proteins_per_taxon=args.per_taxon,
            resume=args.resume,
            taxa_to_scrape=args.taxa
        ))


if __name__ == '__main__':
    main()
