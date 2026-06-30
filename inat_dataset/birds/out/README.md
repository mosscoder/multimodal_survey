---
tags:
  - botany
  - biodiversity
  - image-classification
pretty_name: Sky / cloud hard negatives (bald eagle, in flight)
viewer: false
---

# Sky / cloud hard negatives (bald eagle, in flight)

> **Private training data — not for redistribution.** The shipped product is a
> model; these images are aggregated third-party media used internally under
> their original licenses. Bald eagle (Haliaeetus leucocephalus) iNaturalist images, used as a sky/cloud hard-negative class for a plant detector via upper-corner (sky-maximizing) crops.

- **Taxon:** Haliaeetus leucocephalus (`5305`, species)
- **Sources:** inaturalist
- **Images:** 1003
- **Accessed:** 2026-06-29
- **Split strategy:** `random`

## Classes (species)

| class (label) | images |
|---|---|
| Haliaeetus leucocephalus | 1003 |

## Configurations (one per species)

Each species is a separate configuration with its own train and test split,
loaded with `load_dataset("<repo_id>", "<configuration>")`. New species can be
added later as new configurations without changing the ones already published.

| configuration | splits (images) |
|---|---|
| `haliaeetus_leucocephalus` | test 200, train 800 |

## License breakdown (per image)

| license | images | publicly redistributable |
|---|---|---|
| CC-BY-NC | 863 | no |
| CC-BY | 103 | yes |
| CC0 | 18 | yes |
| CC-BY-NC-SA | 16 | no |
| CC-BY-SA | 3 | yes |

Dataset is PRIVATE training data for a model; non-redistributable licenses are retained and used internally, not republished.

## Contributing datasets / sources

| dataset | source | DOI | images |
|---|---|---|---|
| iNaturalist | inaturalist | — | 1003 |

When publishing, cite these sources (see `bibliography.bib`) and iNaturalist
(www.inaturalist.org), accessed 2026-06-29.

## Top contributors (photographer / collector)

- Pam Hardy (70)
- bev435 (17)
- Elisabeth G (14)
- jojo_potato (11)
- Sean Daniels (11)
- salamanneder (8)
- Justin Dunning (8)
- craigjhowe (7)
- Ryan M. (7)
- Mark Olivier (7)
- Kathlyn Stauffer (6)
- Oohlookitsarabbit (6)
- Lukas Evans (6)
- Karen Skelton (5)
- e-a (5)

## Field schema

Every row is one image with denormalized provenance, spatial, temporal, and
botanical-trait metadata (organization, author, dataset+DOI, collection date,
basis of record = field vs. museum, coordinates + uncertainty, phenology, etc.).
See `metadata.csv` for the full per-image table.
