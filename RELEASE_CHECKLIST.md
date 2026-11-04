# Zenodo release checklist

This repository is a supplementary software and derived-data package for the manuscript *Interpretable Machine Learning Evaluates Differential Therapy Effects in Parkinsonian Gait*. It is **not** a published/accepted-paper record.

## Before publishing a GitHub release

- [ ] Confirm that all three named authors approve the software/data release and metadata.
- [ ] Confirm that participant consent, ethics requirements, institutional data-sharing agreements and source-data conditions permit permanent, worldwide redistribution of every included derived table (including repeat-session pseudonyms and trial-level metrics), under CC BY 4.0.
- [ ] Review the released CSVs for identifiers, distinctive combinations of clinical variables and re-identification risk; do not rely solely on replacing identifiers with `P001`-style codes.
- [ ] Check that no private C3D trajectories, clinical source records, credentials or unpublished third-party materials are included in the GitHub tree **or release assets**. The public Git history must also be considered.
- [ ] Confirm the correct order and spelling of creators, and whether affiliations/ORCIDs should be added to `.zenodo.json`. Do not add unverified identifiers.
- [ ] Confirm the proposed release tag and release note wording. Suggested first tag: `v0.1.0` (prepublication supplementary snapshot).
- [ ] Verify the README commands against a clean Python 3.12 environment, especially deterministic analyses, and compare regenerated results with the released tables. Bayesian refits may not be bit-for-bit identical.
- [ ] Confirm that `.zenodo.json` is valid JSON and review the machine-readable license metadata: the Zenodo GitHub integration uses `mit` for the **software** record while this repository also contains CC BY 4.0 data and documentation. The file-specific licenses in `LICENSE-CODE`, `LICENSE-DATA`, `LICENSE-CONTENT` and README continue to apply. Zenodo supports multiple licenses for mixed-content deposits; review the public Zenodo record and add/display CC BY 4.0 alongside MIT if the integration does not do so automatically.

## Publish and archive

1. Log in to [Zenodo](https://zenodo.org/), connect the GitHub account if necessary, open the **GitHub** section, select **Sync now**, and **enable `archudzik/gaitTherapyBiomarkers`**. Enable it **before** publishing the GitHub release; earlier releases are not archived retroactively.
2. Merge the reviewed metadata PR into `main` so the metadata are in the release's tagged commit.
3. On GitHub, select **Releases → Draft a new release**, choose a new tag such as `v0.1.0` targeting `main`, and set the title to **Prepublication supplementary materials v0.1.0**. Use the suggested notes below. Do not tag a stale commit.
4. Publish the release only after the checklist is signed off. When integration is enabled, a published release triggers Zenodo ingestion and DOI creation; a GitHub **draft** is not the public archived snapshot.
5. In Zenodo, verify creators, title, version, file contents, licensing for both code and data, and the final record DOI. Report any ingestion errors and correct the metadata before making a replacement release.
6. Add the **real** DOI to README and, if appropriate, to the manuscript's availability statement. Do not substitute a speculative DOI, a paper DOI or a Zenodo concept DOI without checking the intended citation.

## Suggested GitHub release notes

**Prepublication supplementary materials for the manuscript “Interpretable Machine Learning Evaluates Differential Therapy Effects in Parkinsonian Gait.”**

This snapshot preserves the author-produced analysis scripts, pseudonymized derived gait and clinical-feature tables, aggregate cohort summary, machine-readable results, and data dictionary used for reproducibility. Raw motion-capture files, linked clinical source records, and third-party source materials are not part of this release.

The original code is licensed under MIT; released derived data, result tables and original documentation are licensed under CC BY 4.0. Consult the repository's separate license files for scope and attribution. Publication details and a manuscript DOI will be added when available.

This is an archival research supplement, not evidence that the manuscript has been accepted or published.
