# Peer Review Report

**Manuscript:** "Detecting Equatorial Plasma Bubbles with Machine Learning"  
**Type:** Systematic Literature Review  
**Reviewer recommendation:** Major Revision  

---

## 1. Overall Assessment

### Novelty and Relevance
The topic is timely and relevant. The intersection of Machine Learning and ionospheric monitoring is a growing research area, and a systematic review consolidating emerging methodologies for EPB detection fills a genuine niche. However, with only 11 included studies—and two of those being tangential to EPBs (MSTIDs and OH* airglow turbulence)—the review's scope is narrow. The novelty is moderate: the paper provides a useful catalogue of techniques but does not offer a meta-analytical synthesis, a novel taxonomic framework, or quantitative cross-study comparisons that would elevate it beyond a descriptive survey.

### Clarity of the Research Objective
The research objective is clearly stated in the Introduction. However, the scope boundary is insufficiently justified. The rationale for including non-EPB phenomena (P2: OH* airglow in Germany; P9: MSTIDs in Japan) on the grounds of "methodological and phenomenological similarities" is weak and potentially undermines the coherence of the review. A reader focused on EPB detection gains little from the analysis of mid-latitude OH* turbulence classification.

### Suitability for Publication
In its current form, the manuscript has significant methodological and structural shortcomings that would likely prevent acceptance at a Q1 journal. With substantial revision, particularly regarding the PRISMA compliance, quality assessment, and analytical depth, the paper could reach an acceptable standard.

---

## 2. Methodology (PRISMA)

### Critical Issues

1. **No PRISMA flow diagram.** This is a mandatory element of any PRISMA-compliant review. The paper must include a standard PRISMA flow diagram showing the exact number of records identified, screened, excluded (with reasons), and included at each stage. Section 2.7 is unacceptably vague—the text says "an extensive pool of preliminary 20 records" (which is not extensive) and then fails to report the exact count after duplicate removal, after title/abstract screening, or after full-text screening with exclusion reasons. This alone is grounds for rejection.

2. **No quality assessment / risk of bias evaluation.** PRISMA requires a critical appraisal of included studies. No standardized tool (e.g., QUADAS-2, Newcastle-Ottawa Scale, or a custom checklist) was applied to assess the methodological quality or risk of bias of the 11 included studies. The review treats all studies as equally valid, which is methodologically unsound. For instance, P6 (Srisamoodkham et al., 2022) is a 7-page conference paper with minimal evaluation, while P3 (Zhong et al., 2025) is a comprehensive journal article—these cannot be weighted equally without justification.

3. **No protocol registration.** Was the review protocol pre-registered (e.g., in PROSPERO or OSF)? This should be stated explicitly. Lack of registration is a transparency concern.

4. **Extremely small initial pool (20 records).** Retrieving only 20 records from three databases for a topic at the intersection of ML and ionospheric science is alarmingly low. This strongly suggests the search strategy is overly restrictive. The mandatory "airglow" group in the search string excludes studies using GNSS-TEC, radar, or satellite in-situ data *without* co-occurring airglow terms, yet such studies are discussed in the review. Consider whether the search string inadvertently filtered out relevant work.

5. **ArXiv inclusion contradicts Inclusion Criterion #1.** The inclusion criteria require "peer-reviewed, original research articles," yet ArXiv is a preprint server with no peer review. This is a direct inconsistency that must be resolved. Either remove ArXiv as a database or modify the inclusion criteria and justify the decision.

6. **Exclusion of IEEE Xplore is poorly justified.** Stating "limitations in search string complexity and inconsistent retrieval" is insufficient. IEEE Xplore is a major source for ML and signal processing literature. Was the same Boolean query tested? How many records were retrieved? The exclusion should be documented with specific evidence or the database should be included.

7. **No date range specified.** The methodology does not state the temporal window of the search. The results reveal all papers are from 2022–2025, but it is unclear whether the search was restricted to this period or whether no earlier papers were found. This must be explicit.

8. **Single-reviewer screening.** The screening was conducted by a single researcher with no inter-rater reliability assessment. This introduces subjective bias. PRISMA guidelines recommend at least two independent reviewers. If a second reviewer was not feasible, this must be acknowledged as a limitation.

### Minor Issues

9. **Section 2.6 formatting.** The two-stage screening description uses asterisks (`*`) instead of proper LaTeX list environments (`\begin{itemize}` / `\begin{enumerate}`), which will render incorrectly.

10. **Typo in label:** `\label{ssubsecec:inclusion-criteria}` contains a double "sec".

11. **Subsection numbering redundancy.** Subsection labels like "2.1. Search Strategy" hardcode the section number, which will conflict with automatic LaTeX numbering (producing "2.1. 2.1. Search Strategy" or similar in the compiled output). Use either manual or automatic numbering, not both.

---

## 3. Results

### Strengths
- The two summary tables (Tables 1 and 2) provide a useful at-a-glance comparison.
- The thematic subsections (problem types, data sources, ML techniques, metrics) organize the information logically.

### Weaknesses

1. **Excessive repetition between tables and narrative text.** Subsections 3.3–3.7 largely re-state information already present in the tables in prose form without adding interpretive value. For example, Section 3.4 ("Data Types and Sources") adds nothing beyond what is already visible in the "Dataset" column of Table 1. The narrative should synthesize, not merely restate.

2. **No visual synthesis.** The paper lacks any figures. A bar chart of techniques used, a timeline of publications, a radar/spider chart of evaluation metrics, or a Venn diagram of data modalities would significantly improve readability and demonstrate synthesis. For a review paper, the absence of any figure is a notable weakness.

3. **P5 and P8 overlap.** Thanakulketsarat et al. (2023) appear twice: P5 is a journal article in *Earth, Planets and Space* and P8 is a conference paper at URSI GASS. Both are by the same first author, use the same dataset (VHF radar from Chumphon, Thailand), the same method (CNN-SVM), and report nearly identical accuracies (93.67% vs. 93.08%). The paper should acknowledge this overlap explicitly and justify including both. Counting them as two independent studies inflates the corpus size.

4. **The Siddiqui (P7) reference is incomplete.** The bibliography entry `siddiquiENHANCEDIDENTIFICATIONEQUATORIAL` lacks a year, journal name, volume, pages, and DOI. It appears to be an unpublished or grey-literature source. If it is not peer-reviewed, including it violates Inclusion Criterion #1. Despite this, it is one of the most cited references in the Discussion (cited 8+ times). The provenance and peer-review status of this source must be clarified.

5. **Table readability.** Both tables use `\tiny` font size. At compilation, these will be nearly illegible. Consider splitting Table 2 or rotating to landscape orientation.

---

## 4. Discussion

### Strengths
- The discussion is well-structured with clearly delineated subsections covering technique comparison, data modalities, limitations, challenges, gaps, and future work.
- The identification of the detection-vs-prediction gap and the absence of multimodal fusion frameworks are valuable contributions.

### Weaknesses

1. **Lack of a "Limitations of this review" subsection.** The Discussion identifies limitations of the *reviewed studies* but never critically examines the limitations of *the review itself*. This is a major omission. Self-critique should address: the small corpus size (11 studies), single-reviewer bias, narrow search strategy, inclusion of non-EPB studies, absence of quality assessment, and the geographic skew of included studies.

2. **Over-reliance on a single reference.** Sections 4.5, 4.6, and 4.7 collectively cite `\cite{siddiquiENHANCEDIDENTIFICATIONEQUATORIAL}` at least 6 times, often as the sole supporting reference. Entire subsections (4.6 and 4.7) are supported exclusively by this one source. This is problematic for two reasons: (a) it weakens the cross-study synthesis claim, and (b) the reference itself appears to lack basic bibliographic metadata. A systematic review should derive its conclusions from patterns across the corpus, not from a single paper's assertions.

3. **Descriptive rather than critical in places.** Section 4.3 (Impact of Different Data Types) reads as a catalogue of instrument characteristics rather than a critical analysis derived from the reviewed studies. What did the *evidence from the 11 studies* specifically reveal about data modality trade-offs? Which studies attempted cross-instrument validation, and what were the quantitative outcomes?

4. **No quantitative cross-study comparison.** The discussion would benefit significantly from a structured comparison—e.g., do hybrid models consistently outperform pure DCNN models when controlling for dataset size? How does reported accuracy correlate with dataset size or class imbalance ratio? Even a qualitative comparison table would add value.

5. **Typo:** Section 4.4 contains "fluid fluid transitions" (duplicated word).

6. **Subsection numbering.** Same issue as in Methodology—hardcoded "4.1.", "4.2." etc. will conflict with LaTeX automatic numbering.

---

## 5. Writing Quality

### Clarity and Tone
The academic tone is generally appropriate and the writing is competent. Sentences are grammatically correct and terminology is used consistently.

### Redundancy
This is the paper's most significant writing flaw. The same points are repeated across multiple sections:

- The susceptibility to cloud cover is mentioned in the Abstract, Introduction, Results (Table 2), Discussion §4.4, and Conclusion.
- The dataset imbalance problem appears in the Abstract, Results (Table 2), Discussion §4.4, and Conclusion.
- The future work on multimodal fusion and predictive modeling is stated nearly identically in Discussion §4.7 and the Conclusion.
- The XAI findings about vertical extent and plume width are described in Results §3.7, Discussion §4.2, and the Conclusion.

Each section should have a distinct purpose. The Results should *report*, the Discussion should *interpret*, and the Conclusion should *summarize* without re-arguing.

### Logical Flow
The transition from Results to Discussion is weak. The Results section ends with interpretive statements (e.g., "high solar activity promotes EPB area, poleward latitudinal extension...") that belong in the Discussion. Conversely, the Discussion sometimes re-introduces factual findings. The boundary between reporting and interpreting should be sharpened.

### Conclusion
The Conclusion is a single dense paragraph that reads as an abbreviated combination of the Discussion's subsections. It would benefit from being more concise and forward-looking, avoiding the repetition of specific technical details already covered.

---

## 6. Strengths

1. **Timely topic** with clear practical relevance for space weather monitoring.
2. **Well-structured Discussion** with logical subsections that progress from findings to limitations to future directions.
3. **Comprehensive tables** that consolidate objectives, techniques, metrics, results, and limitations in a single comparative view.
4. **Strong search string design.** The three-group Boolean structure is well thought out and captures the conceptual intersection effectively.
5. **Identification of literature gaps** (predictive modeling, multimodal fusion) is a useful contribution for the community.
6. **XAI discussion** linking ML interpretability to physical validation is a valuable angle not commonly seen in space weather reviews.

---

## 7. Weaknesses and Risks of Rejection

1. **Non-compliance with PRISMA:** No flow diagram, no quality assessment, vague screening numbers. This is the single most critical flaw for a paper that claims PRISMA adherence.
2. **Very small corpus (11 studies, effectively 9–10 unique).** This limits the generalizability of any conclusions drawn and makes the paper vulnerable to the criticism of being premature.
3. **Internal inconsistencies in methodology** (ArXiv vs. peer-review criterion; IEEE exclusion; potential inclusion of non-peer-reviewed work).
4. **Excessive redundancy** across sections, reducing the manuscript's information density.
5. **Absence of figures** in a review paper.
6. **Over-reliance on a single, incompletely cited reference** (Siddiqui) for the most important Discussion claims.
7. **No self-critical limitations section** for the review itself.
8. **Two near-duplicate studies (P5/P8)** counted separately without acknowledgment.

---

## 8. Specific Suggestions for Improvement

### High Priority (required before resubmission)

| # | Suggestion |
|---|-----------|
| 1 | **Add a PRISMA flow diagram** with exact counts at each stage (identification, screening, eligibility, inclusion) and specific exclusion reasons. |
| 2 | **Conduct and report a quality assessment** of all 11 included studies using a recognized appraisal tool. Report results in a dedicated table. |
| 3 | **Resolve the ArXiv/peer-review inconsistency.** Either exclude ArXiv, add a non-peer-reviewed category, or adjust the inclusion criteria with justification. |
| 4 | **Verify and complete the Siddiqui (P7) reference.** Provide year, journal, volume, pages, and DOI. If it is not peer-reviewed, reconsider its inclusion or at minimum flag it explicitly. |
| 5 | **Report exact numbers for every screening stage** in Section 2.7: records per database, duplicates removed, title/abstract exclusions, full-text exclusions with reasons. |
| 6 | **Add a "Limitations of this Review" subsection** in the Discussion addressing corpus size, single-reviewer screening, search limitations, and scope boundary decisions. |
| 7 | **Acknowledge and discuss the P5/P8 overlap** (same group, same method, same dataset, same conference/journal pair). |

### Medium Priority (strongly recommended)

| # | Suggestion |
|---|-----------|
| 8 | **Add at least 2–3 figures:** e.g., PRISMA flow diagram, a summary chart of ML techniques vs. performance, a publication timeline. |
| 9 | **Reduce redundancy.** Perform a pass where each fact is assigned to exactly one section. The Conclusion should be at most 8–10 sentences summarizing key points, not re-arguing the Discussion. |
| 10 | **Diversify citations in Discussion §4.5–4.7.** Derive gaps and future directions from cross-study patterns rather than a single source. |
| 11 | **Sharpen the Results/Discussion boundary.** Move all interpretive statements from Results §3.7 to the Discussion. Keep Results purely descriptive. |
| 12 | **Justify the inclusion of P2 and P9** more rigorously, or consider removing them to tighten the review's focus. |
| 13 | **Specify the search date range** and the exact date the search was executed. |
| 14 | **Broaden the search** if possible—consider Google Scholar, PubMed, or re-attempting IEEE Xplore to increase the initial pool beyond 20 records. |

### Low Priority (polish)

| # | Suggestion |
|---|-----------|
| 15 | Fix subsection numbering: remove hardcoded numbers ("2.1.", "4.1." etc.) to avoid conflicts with LaTeX automatic numbering. |
| 16 | Fix the label typo `\label{ssubsecec:inclusion-criteria}`. |
| 17 | Fix `* \textbf{Stage 1...}` formatting in Section 2.6 to use proper `\begin{itemize}`. |
| 18 | Fix "fluid fluid transitions" → "fluid transitions" in Section 4.4. |
| 19 | Improve table legibility: use `\footnotesize` instead of `\tiny`, or switch to landscape orientation. |
| 20 | Ensure inclusion/exclusion criteria are formatted as proper numbered lists rather than inline sentences. |

---

## 9. Final Recommendation

**Decision: Major Revision**

### Justification

The manuscript addresses a relevant and timely research question, and its structured Discussion provides genuinely useful observations for the space weather ML community—particularly regarding the detection-vs-prediction gap, model generalizability challenges, and the potential of XAI for physical discovery. These are commendable contributions.

However, the paper cannot be accepted in its current form due to several fundamental shortcomings:

1. The claimed PRISMA compliance is incomplete. The absence of a flow diagram, quality assessment, and transparent screening counts undermines the methodological rigor that is the primary value proposition of a systematic review.
2. The corpus of 11 studies (with two near-duplicates and two tangential inclusions) is small, and the initial retrieval of only 20 records raises concerns about search comprehensiveness.
3. Critical Discussion claims rest disproportionately on a single, incompletely cited reference.
4. Significant redundancy across sections reduces the manuscript's analytical density.

None of these issues are insurmountable. With a thorough revision addressing the high-priority suggestions above, the manuscript has the potential to make a meaningful contribution to the literature. The authors are encouraged to revise and resubmit.
