---
tags:
  - botany
  - biodiversity
  - image-classification
pretty_name: Montana native and invasive plants (50 species)
viewer: false
---

# Montana native and invasive plants (50 species)

> **Private training data — not for redistribution.** The shipped product is a
> model; these images are aggregated third-party media used internally under
> their original licenses. A set of 50 plant species of western Montana rangeland, spanning native forbs and grasses and invasive forbs and grasses, harvested from iNaturalist for a species level image classifier. Each species is its own configuration with its own train and test split. Candidates are sampled month-balanced across the year (not newest-first) for even seasonal coverage.

- **Classes:** 50 species (see below)
- **Sources:** inaturalist
- **Images:** 48011
- **Accessed:** 2026-07-01
- **Split strategy:** `random`

## Classes (species)

| class (label) | images |
|---|---|
| Ratibida columnifera | 1015 |
| Erigeron speciosus | 1005 |
| Grindelia squarrosa | 1004 |
| Helianthus annuus | 1003 |
| Collomia linearis | 1003 |
| Lithospermum ruderale | 1003 |
| Artemisia frigida | 1003 |
| Lomatium triternatum | 1002 |
| Phacelia hastata | 1002 |
| Hesperostipa comata | 1002 |
| Sisymbrium altissimum | 1002 |
| Balsamorhiza sagittata | 1001 |
| Lupinus sericeus | 1001 |
| Gaillardia aristata | 1001 |
| Achillea millefolium | 1001 |
| Lithophragma glabrum | 1001 |
| Microsteris gracilis | 1001 |
| Phacelia linearis | 1001 |
| Cirsium undulatum | 1001 |
| Collinsia parviflora | 1001 |
| Erigeron pumilus | 1001 |
| Plantago patagonica | 1001 |
| Eriophyllum lanatum | 1001 |
| Rosa woodsii | 1001 |
| Bromus tectorum | 1001 |
| Poa bulbosa | 1001 |
| Taraxacum officinale | 1001 |
| Tragopogon dubius | 1001 |
| Verbascum thapsus | 1001 |
| Chenopodium album | 1001 |
| Salsola tragus | 1001 |
| Silene latifolia | 1001 |
| Helianthus maximiliani | 1000 |
| Myosotis stricta | 1000 |
| Linum lewisii | 1000 |
| Artemisia tridentata | 1000 |
| Pseudoroegneria spicata | 1000 |
| Thinopyrum intermedium | 1000 |
| Poa compressa | 1000 |
| Euphorbia virgata | 1000 |
| Holosteum umbellatum | 1000 |
| Verbascum blattaria | 1000 |
| Veronica verna | 1000 |
| Filago arvensis | 1000 |
| Bassia scoparia | 1000 |
| Antennaria parvifolia | 977 |
| Poa secunda | 806 |
| Festuca idahoensis | 464 |
| Helianthella uniflora | 440 |
| Polygonum douglasii | 259 |

## Configurations (one per species)

Each species is a separate configuration with its own train and test split,
loaded with `load_dataset("<repo_id>", "<configuration>")`. New species can be
added later as new configurations without changing the ones already published.

| configuration | splits (images) |
|---|---|
| `balsamorhiza_sagittata` | test 200, train 800 |
| `lupinus_sericeus` | test 200, train 800 |
| `gaillardia_aristata` | test 200, train 800 |
| `achillea_millefolium` | test 200, train 800 |
| `helianthus_annuus` | test 200, train 800 |
| `helianthus_maximiliani` | test 200, train 800 |
| `antennaria_parvifolia` | test 196, train 781 |
| `collomia_linearis` | test 200, train 800 |
| `lithophragma_glabrum` | test 200, train 800 |
| `lomatium_triternatum` | test 200, train 800 |
| `microsteris_gracilis` | test 200, train 800 |
| `myosotis_stricta` | test 200, train 800 |
| `phacelia_linearis` | test 200, train 800 |
| `lithospermum_ruderale` | test 200, train 800 |
| `cirsium_undulatum` | test 200, train 800 |
| `collinsia_parviflora` | test 200, train 800 |
| `erigeron_pumilus` | test 200, train 800 |
| `helianthella_uniflora` | test 88, train 350 |
| `linum_lewisii` | test 200, train 800 |
| `plantago_patagonica` | test 200, train 800 |
| `polygonum_douglasii` | test 52, train 206 |
| `artemisia_frigida` | test 200, train 800 |
| `artemisia_tridentata` | test 200, train 800 |
| `erigeron_speciosus` | test 200, train 800 |
| `eriophyllum_lanatum` | test 200, train 800 |
| `grindelia_squarrosa` | test 200, train 800 |
| `phacelia_hastata` | test 200, train 800 |
| `ratibida_columnifera` | test 200, train 800 |
| `rosa_woodsii` | test 200, train 800 |
| `poa_secunda` | test 161, train 644 |
| `pseudoroegneria_spicata` | test 200, train 800 |
| `hesperostipa_comata` | test 200, train 800 |
| `festuca_idahoensis` | test 93, train 371 |
| `thinopyrum_intermedium` | test 200, train 800 |
| `bromus_tectorum` | test 200, train 800 |
| `poa_bulbosa` | test 200, train 800 |
| `poa_compressa` | test 200, train 800 |
| `euphorbia_virgata` | test 200, train 800 |
| `sisymbrium_altissimum` | test 200, train 800 |
| `holosteum_umbellatum` | test 200, train 800 |
| `taraxacum_officinale` | test 200, train 800 |
| `tragopogon_dubius` | test 200, train 800 |
| `verbascum_blattaria` | test 200, train 800 |
| `verbascum_thapsus` | test 200, train 800 |
| `veronica_verna` | test 200, train 800 |
| `chenopodium_album` | test 200, train 800 |
| `filago_arvensis` | test 200, train 800 |
| `bassia_scoparia` | test 200, train 800 |
| `salsola_tragus` | test 200, train 800 |
| `silene_latifolia` | test 200, train 800 |

## License breakdown (per image)

| license | images | publicly redistributable |
|---|---|---|
| CC-BY-NC | 39799 | no |
| CC-BY | 5456 | yes |
| CC0 | 2162 | yes |
| CC-BY-NC-SA | 443 | no |
| CC-BY-SA | 151 | yes |

Dataset is PRIVATE training data for a model; non-redistributable licenses are retained and used internally, not republished.

## Contributing datasets / sources

| dataset | source | DOI | images |
|---|---|---|---|
| iNaturalist | inaturalist | — | 48011 |

When publishing, cite these sources (see `bibliography.bib`) and iNaturalist
(www.inaturalist.org), accessed 2026-07-01.

## Top contributors (photographer / collector)

- Norbert Kondla (1497)
- Elora (466)
- John D Reynolds (464)
- Z (403)
- Jack Bindernagel (401)
- Cassidy Best (395)
- Justin Chan (246)
- soda (245)
- Abby Hyde (234)
- Daryl Nolan (226)
- Alexey P. Seregin (217)
- Kallum McDonald (200)
- John Gaskin (191)
- Daughter Dad (176)
- Finn McGhee (176)

## Field schema

Every row is one image with denormalized provenance, spatial, temporal, and
botanical-trait metadata (organization, author, dataset+DOI, collection date,
basis of record = field vs. museum, coordinates + uncertainty, phenology, etc.).
See `metadata.csv` for the full per-image table.
