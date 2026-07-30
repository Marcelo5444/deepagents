# Literature Review Skill

## Overview
Guidance for effective scientific literature retrieval and synthesis using
arXiv, Semantic Scholar, and PubMed.

## When to Use
- Searching for papers on a specific topic
- Finding papers by DOI/PMID/arXiv ID
- Identifying key papers in a field
- Tracking citations and related work

## Search Strategies

### arXiv
- Use category prefixes: `cat:cond-mat.mtrl-sci`, `cat:physics.chem-ph`
- Date ranges: `submittedDate:[20240101 TO 20241231]`
- Combine terms: `all:"solid-state battery" AND cat:cond-mat.mtrl-sci`

### Semantic Scholar
- Search by DOI: `DOI:10.1038/s41586-023-06000-9`
- Filter by year: `year:2023-2024`
- Filter by venue: `venue:"Nature"`
- Use citation count for impact: `citationCount:>100`

### PubMed
- MeSH terms: `"Hydrogels"[Mesh] AND "Drug Delivery"[Mesh]`
- Date filters: `("2024/01/01"[Date - Publication] : "2024/12/31"[Date - Publication])`
- Article types: `review[pt]`, `clinical trial[pt]`

## Best Practices

1. **Start broad, then narrow**
   - First search: general terms
   - Refine: add specific materials, methods, years

2. **Use multiple sources**
   - arXiv for latest preprints
   - Semantic Scholar for citations and metadata
   - PubMed for biomedical focus

3. **Verify claims**
   - Always cite DOI/PMID/arXiv ID
   - Check citation count for credibility
   - Look for replication studies

4. **Track search history**
   - Log queries and results
   - Note which papers were most relevant
   - Save PDFs for key papers

## Common Tasks

### Find papers on a material property
```
Search: "Li-rich cathode voltage fade mechanism"
Sources: Semantic Scholar, arXiv
Filter: 2022-2024, materials science
```

### Find computational method details
```
Search: "DFT+U VASP Li1.2Ni0.13Mn0.54Co0.13O2 oxygen redox"
Sources: Semantic Scholar
Filter: 2020-2024, physics/chemistry
```

### Track citations of a key paper
```
1. Get paper details by DOI
2. Use citations field to find citing papers
3. Filter by year and relevance
```

## Output Format
When reporting literature findings, always include:
- Paper title
- Authors
- DOI / PMID / arXiv ID
- Year
- Key finding relevant to query
- Citation count (if available)