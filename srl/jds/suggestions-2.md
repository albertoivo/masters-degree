# Peer Review Report (Round 2)

**Manuscript:** "Detecting Equatorial Plasma Bubbles with Machine Learning"  
**Type:** Systematic Literature Review  
**Journal class:** jdsart (Journal of Data Science)  
**Reviewer recommendation:** Major Revision  

---

## 1. Overall Assessment

### Novelty and Relevance
The manuscript addresses a timely niche: the application of ML to automated EPB detection. Given the exponential growth of ionospheric observation data and the unsustainability of manual inspection, a systematic consolidation of emerging computational approaches fills a genuine gap. However, the review's contribution remains primarily cataloguing rather than analytical. The paper does not propose a novel taxonomy, a quantitative meta-comparison, or a standardized benchmarking framework. With only 9 included studies spanning 2022–2025, the review captures a nascent field—this is acknowledged—but the small corpus places a high burden on the quality of synthesis to justify publication.

### Clarity of the Research Objective
The research objective is stated clearly in the final paragraph of the Introduction: synthesize computational approaches, datasets, and evaluation metrics, identify methodological patterns, and uncover gaps. This is appropriate for a systematic review.

### Suitability for Publication
The manuscript has improved structurally (PRISMA diagram, dual reviewers, improved synthesis in Results, reduced redundancy), but several fundamental methodological shortcomings remain that would prevent acceptance at a Q1 venue without further revision.

---

## 2. Methodology (PRISMA)

### 2.1 Search Strategy

**Strengths:**
- Three databases (Scopus, Web of Science, IEEE Xplore) are appropriate for this interdisciplinary topic.
- The three-group Boolean search string is logically constructed and captures the conceptual intersection of EPBs, optical instrumentation, and ML techniques.
- The statement that no date restriction was applied, with the 2022–2025 coverage reflecting the field's emergence, is a welcome clarification.

**Weaknesses:**

1. **Search string excludes relevant data modalities.** The mandatory middle group (`"airglow" OR "all-sky imag*" OR "nightglow" OR "OI 630"`) restricts retrieval to optical instrument terminology. Yet the review includes studies based on VHF radar (P4, P7) and GNSS (P1). These studies could only have been captured if their abstracts happened to mention airglow-related terms. This creates a systematic retrieval bias: studies using ML on radar or GNSS data for EPB detection that do *not* mention airglow in their title/abstract/keywords would be missed. The search string and the actual inclusion scope are misaligned. This must be either (a) acknowledged as a limitation, or (b) the search string should be broadened to include radar and GNSS terms, and the search re-run.

2. **No search execution date reported.** PRISMA requires reporting the date the search was last executed. This is absent.

3. **No protocol registration.** Was the review protocol pre-registered (e.g., PROSPERO, OSF)? This is not mentioned. While not always mandatory, the absence should be disclosed and justified.

4. **37 records is still low.** While improved from the earlier 20, retrieving 37 records from three major databases for a topic at the intersection of ML and ionospheric science suggests the search may still be overly restrictive (see point 1). For context, a broader search without the optical-instrument constraint might yield substantially more candidates.

### 2.2 Inclusion and Exclusion Criteria

5. **Criteria are presented as inline numbered sentences** rather than structured lists. For a methodology section, proper `\begin{enumerate}` formatting would improve clarity and allow precise referencing during screening documentation.

6. **Inclusion Criterion 1 ("Peer-reviewed, original research articles") vs. actual inclusions.** Two concerns:
   - **P5 (Srisamoodkham et al., 2022)** is a conference proceedings paper (Springer LNNS, 7 pages). Depending on the journal's definition, conference papers may or may not qualify as "peer-reviewed articles." The decision to include it should be explicitly justified.
   - **P6 (Siddiqui, 2025)** is published in *International Journal of Advanced Artificial Intelligence Research (IJAAIR)*, Vol. 2, No. 2, pp. 01–08. This journal has no impact factor, no indexing in Scopus or Web of Science, launched very recently, and exhibits multiple hallmarks of a predatory or extremely low-quality venue (single-digit page articles, no identifiable editorial board of repute, DOI prefix 10.55640). Including a study from such a source without any quality stratification is methodologically problematic—especially given that this reference is cited more than any other in the Discussion (appearing in §4.2, §4.3, §4.4, §4.5, §4.6, and §4.7). **This is a critical issue that must be addressed.** At minimum, the peer-review provenance and journal quality of P6 should be discussed transparently.

### 2.3 Study Selection Process

7. **Dual-reviewer screening is now reported—good.** The statement that "both reviewers reached full agreement" is noted but somewhat unusual. Perfect inter-rater agreement across all stages, with no conflicts requiring resolution, would benefit from a quantitative measure (e.g., Cohen's kappa or percent agreement) to be fully credible.

8. **Section 2.7 remains vague on stage-specific numbers.** The text states "37 records" initially and "9 articles" finally, but does not report: (a) records per database before deduplication, (b) number of duplicates removed, (c) number excluded at title/abstract screening with reasons, (d) number excluded at full-text screening with reasons. The PRISMA flow diagram figure presumably contains some of this information, but the narrative text should also provide these numbers explicitly. A reader should not have to rely solely on a figure for critical methodological data.

### 2.4 Quality Assessment

9. **No quality assessment or risk-of-bias evaluation.** This remains the single most critical methodological omission. PRISMA 2020 (Item 13) requires authors to describe methods used for assessing risk of bias of included studies.  No standardized appraisal tool (QUADAS-2, Newcastle-Ottawa, JBI checklist, or a custom instrument) has been applied. The review treats all 9 studies—ranging from a 7-page conference proceeding (P5) to comprehensive journal articles (P2)—as equally valid evidence. A quality assessment table is essential and its absence alone may be grounds for rejection at rigorous venues.

---

## 3. Results

### 3.1 Tables

10. **Tables use `\tiny` font size.** At compilation, both Table 1 and Table 2 will be extremely difficult to read. Switching to `\footnotesize` or `\scriptsize`, potentially combined with landscape orientation for Table 2, would significantly improve legibility.

11. **Table content is generally informative.** The parallel structure (Table 1 for objectives/data, Table 2 for techniques/results/limitations) is well designed and serves the reader effectively.

### 3.2 Synthesis Quality

The Results subsections (§3.2–§3.7) show meaningful improvement over a purely descriptive listing. The identification of three methodological paradigms (§3.5), the observation about the two-stage detection+classification workflow (§3.3), and the critique of accuracy as a standalone metric in imbalanced settings (§3.6) are genuine analytical contributions.

However:

12. **Section 3.7 contains interpretive statements that belong in the Discussion.** For example: "Two cross-cutting findings are noteworthy" followed by interpretive synthesis of XAI implications and solar-cycle profiling. The Results section should present findings; interpretation should be reserved for the Discussion. The boundary remains blurred.

13. **P4 and P7 overlap is not acknowledged.** Thanakulketsarat et al. (2023) appear as both P4 (journal: *Earth, Planets and Space*, 93.67% accuracy) and P7 (conference: *URSI GASS 2023*, 93.08% accuracy). Both use the same first author, same research group, same dataset (VHF radar, Thailand), and nearly identical CNN-SVM methodology. P7's bibliography entry (`thanakulketsarat2023equatorial`) lacks volume, pages, and DOI—it is a conference abstract. Counting these as two independent studies inflates the corpus from effectively 8 to 9 unique investigations. The paper must:
    - Acknowledge the overlap explicitly.
    - Justify including both (e.g., if P7 reports different experimental conditions).
    - Alternatively, merge them into a single entry and adjust all counts.

14. **No sensitivity or subgroup analysis.** With 9 studies, formal meta-analysis is not expected, but even a qualitative subgroup comparison would strengthen the synthesis. For example: do studies using ASI data consistently outperform those using radar? How does dataset size correlate with reported accuracy? A simple table cross-referencing key design choices with performance outcomes would add analytical value.

### 3.3 Figure

15. **The bar chart (Figure 2) is a welcome addition,** but a single figure for a review paper is still minimal. Consider adding: (a) a publication timeline showing growth of the field, (b) a comparative performance chart (e.g., accuracy vs. method type), or (c) a geographic map of study locations. Visual synthesis is a hallmark of high-quality review papers.

---

## 4. Discussion

### 4.1 Structure and Coverage

The Discussion is well-organized into seven thematic subsections progressing logically from interpretation → technique comparison → data modalities → limitations → challenges → gaps → future work. This structure is the paper's strongest feature.

### 4.2 Critical Analysis

16. **Section 4.1 (Interpretation)** provides a competent high-level summary but would benefit from more specificity. The claim that ML models "identify subtle spatial gradients, textures, and intensity variations that are often visually ambiguous to the human eye" is supported only by \cite{okoh2025bootstrapping}. Is there quantitative evidence from the reviewed studies (e.g., cases where ML detected EPBs that human annotators missed)? If not, this is an aspirational assertion rather than an evidence-based finding.

17. **Section 4.2 (Technique Comparison)** is the strongest analytical subsection. The complexity-vs-efficiency trade-off is well articulated, and the discussion of ensemble methods and XAI integration is informative. However, the claim that hybrid models achieve "competitive accuracies" while reducing model size "by up to 70%" is derived from a single study (P3/Yacoub). Presenting a finding from one study as a generalizable pattern of hybrid approaches is misleading.

18. **Section 4.3 (Data Types)** reads more as an instrument capabilities overview than a synthesis of evidence from the reviewed studies. The paragraph on VHF radar advantages and GNSS coverage is general domain knowledge, not a finding derived from the systematic review. This subsection should focus on what the *included studies* revealed about data modality trade-offs—e.g., did radar-based studies (P4, P7) achieve different accuracy ranges than ASI-based studies?

19. **Over-reliance on \cite{siddiqui2025enhanced}.** This reference appears as the sole or primary citation in Sections 4.4 (twice), 4.5 (twice), 4.6 (three times), and 4.7 (three times). Sections 4.6 and 4.7 cite *only* this source. Given the concerns about this journal's credibility (see Point 6), anchoring the review's gap analysis and future work agenda on a single reference from a questionable venue severely undermines the Discussion's authority. Gaps and future directions should be derived from cross-study patterns or corroborated by established literature from the ionospheric science community.

### 4.3 Missing Elements

20. **No "Limitations of this Review" subsection.** The Discussion critiques the limitations of the *reviewed studies* thoroughly but never examines the limitations of *the review itself*. This is a mandatory component for any systematic review. At minimum, address:
    - Small corpus size (9 studies) and its implications for generalizability of conclusions.
    - Search string bias toward optical terminology (excluding pure radar/GNSS ML studies).
    - Absence of quality assessment.
    - Geographic and temporal concentration of the literature.
    - Potential publication bias (only positive-result studies captured).
    - The P4/P7 overlap issue.

21. **No discussion of publication bias.** Were all 9 studies reporting favorable results? Were any negative or null findings reported? The absence of a funnel plot or qualitative assessment of publication bias is a gap.

---

## 5. Writing Quality

### 5.1 Clarity and Tone
The writing is grammatically sound, the academic register is consistent, and terminology is used precisely. The manuscript reads fluently, which is a notable strength.

### 5.2 Redundancy

Redundancy has been reduced compared to what one might expect, but residual repetition remains:

- **Environmental contamination / cloud cover** appears in: Introduction (¶2), Results Table 2 (P3 limitation), Discussion §4.4 (¶1), and Conclusion. Four mentions of the same point.
- **Dataset imbalance** appears in: Introduction (¶4), Results Table 2 (P2/P8), Discussion §4.4 (¶2), and Conclusion. Again four mentions.
- **Multimodal fusion as a gap** appears in: Discussion §4.6 and §4.7, and Conclusion. The Conclusion's reference to "limitations discussed herein" is appropriately brief, but §4.6 and §4.7 overlap substantially—both identify the same gap (absence of multi-instrument fusion) from slightly different angles.

**Recommendation:** Each fact should have one authoritative location. The Introduction should *motivate*, the Results should *report*, the Discussion should *interpret*, and the Conclusion should *summarize by reference*. Consider whether the Introduction's mention of "model bias due to solar-cycle data imbalances" (¶4) is premature—this is a finding, not a motivational framing element.

### 5.3 Logical Flow

22. The transition from Results to Discussion is functional but could be strengthened with a brief bridging paragraph at the start of Section 4 that explicitly states how the Discussion will extend beyond the descriptive findings.

23. **Section 4.3 (Data Types) partially overlaps with Results §3.4 (Data Types and Sources).** Both discuss ASI dominance, geographic distribution, and VHF/GNSS roles. The Results section already establishes these facts; the Discussion should focus exclusively on their *implications* (e.g., what does the ASI concentration mean for model generalizability?).

---

## 6. Strengths

1. **Well-defined and relevant research question** filling a genuine gap in the EPB literature.
2. **Strong Discussion structure** (§4.1–§4.7) with logical progression from findings to gaps to future directions.
3. **Improved synthesis in Results** — the three-paradigm framework (end-to-end DL, hybrid, XAI-augmented) is a useful organizational contribution.
4. **PRISMA flow diagram now included.**
5. **Dual-reviewer screening** enhances methodological credibility.
6. **Identification of the detection-vs-prediction gap** and the absence of multimodal fusion frameworks are valuable observations for the community.
7. **XAI discussion** (§4.2) linking interpretability to physical validation is an uncommon and valuable angle in space weather reviews.
8. **Concise Conclusion** that avoids re-arguing the Discussion in detail.
9. **Competent, grammatically sound writing** throughout.

---

## 7. Weaknesses and Risks of Rejection

| # | Weakness | Severity |
|---|----------|----------|
| W1 | **No quality assessment / risk-of-bias evaluation** of included studies. PRISMA non-compliance. | Critical |
| W2 | **Inclusion of P6 (Siddiqui, 2025) from a potentially predatory journal**, with no quality stratification, while it serves as the most-cited reference in the Discussion. | Critical |
| W3 | **P4/P7 overlap** (same group, method, dataset) not acknowledged, inflating corpus to 9 from effectively 8 unique studies. | High |
| W4 | **No "Limitations of this Review" subsection.** | High |
| W5 | **Search string excludes radar/GNSS terminology**, creating a systematic retrieval bias that is not discussed. | High |
| W6 | **Over-reliance on a single questionable reference** (Siddiqui) for gap analysis and future work claims. | High |
| W7 | **No search execution date** reported. | Medium |
| W8 | **No protocol registration** disclosed. | Medium |
| W9 | **Vague screening stage numbers** — exact counts per database and per screening phase not in narrative text. | Medium |
| W10 | **Very small corpus** (9 studies, effectively 8 unique) limits generalizability of conclusions. | Medium |
| W11 | **Section 4.3** is descriptive instrument overview rather than evidence-derived synthesis. | Medium |
| W12 | **Tables use `\tiny` font**, risking illegibility. | Low |
| W13 | **Inclusion/exclusion criteria not formatted as proper lists.** | Low |
| W14 | **Only one figure** besides the PRISMA diagram for a review paper. | Low |

---

## 8. Specific Suggestions for Improvement

### Critical Priority (likely required for acceptance)

| # | Suggestion |
|---|-----------|
| S1 | **Conduct and report a formal quality assessment** of all 9 included studies using a recognized appraisal tool (e.g., a modified JBI checklist for prevalence/analytical studies, or a custom checklist). Present results in a dedicated table. Discuss how study quality affects the confidence in the review's conclusions. |
| S2 | **Address the P6 (Siddiqui) provenance issue.** Investigate the journal (*IJAAIR*) and either: (a) provide evidence it is legitimately peer-reviewed and indexed, (b) flag it explicitly as a low-confidence source and reduce reliance on it in the Discussion, or (c) exclude it if it fails the peer-review inclusion criterion. |
| S3 | **Add a "Limitations of this Review" subsection** (§4.8 or similar) addressing: small corpus, search string optical bias, absence of quality assessment (if not added), geographic concentration, P4/P7 overlap, and potential publication bias. |
| S4 | **Acknowledge and discuss the P4/P7 overlap.** State explicitly that both entries originate from the same research group with near-identical methodology and results. Justify their separate inclusion or merge them. |
| S5 | **Diversify citations in Discussion §4.5–§4.7.** Derive future directions from patterns observed across multiple reviewed studies, not from a single source. Where only one study supports a claim, state this explicitly ("Only P6 identified...") rather than presenting it as a consensus finding. |

### High Priority (strongly recommended)

| # | Suggestion |
|---|-----------|
| S6 | **Report the search execution date** (exact date the last search was run). |
| S7 | **State whether the protocol was pre-registered** (PROSPERO, OSF, or not registered). If not registered, acknowledge this as a limitation. |
| S8 | **Provide exact stage-specific numbers in §2.7 narrative text**: records from Scopus (n=?), Web of Science (n=?), IEEE Xplore (n=?); duplicates removed (n=?); excluded at title/abstract (n=?, with summary reasons); excluded at full-text (n=?, with specific reasons); final included (n=9). The PRISMA figure should complement, not replace, the narrative. |
| S9 | **Acknowledge the search string's optical bias** — the mandatory airglow/ASI group may exclude ML studies on EPBs using purely GNSS or radar data that do not mention optical terms. Discuss how this affects the review's completeness. |
| S10 | **Move interpretive content from Results §3.7** ("Two cross-cutting findings are noteworthy...") to the Discussion. Keep Results factual. |
| S11 | **Rewrite Discussion §4.3** to focus on evidence from the reviewed studies (e.g., accuracy differences between ASI-based and radar-based studies) rather than general instrument characteristics available in textbooks. |

### Medium Priority (recommended for strengthening)

| # | Suggestion |
|---|-----------|
| S12 | **Add a qualitative cross-study comparison table** in either Results or Discussion: rows = studies, columns = key design dimensions (data modality, ML paradigm, dataset size, class balance, best metric), with a summary row identifying patterns. |
| S13 | **Add 1–2 additional figures** for visual synthesis: e.g., a comparative accuracy chart by method type, or a geographic map of study sites. |
| S14 | **Reduce remaining redundancy**: assign each finding (cloud cover limitation, dataset imbalance, multimodal fusion gap) to one authoritative section and reference it elsewhere rather than restating it. |
| S15 | **Format inclusion/exclusion criteria** as proper `\begin{enumerate}` lists rather than inline numbered sentences. |
| S16 | **Increase table font size** from `\tiny` to `\footnotesize` or `\scriptsize`. Consider landscape orientation for Table 2 if space is insufficient. |

### Low Priority (polish)

| # | Suggestion |
|---|-----------|
| S17 | Add a brief bridging sentence at the start of §4 connecting the Results to the Discussion's analytical aims. |
| S18 | In the Introduction ¶4, consider whether mentioning specific findings (solar-cycle imbalance, weather degradation) is appropriate or whether these should be reserved for Results/Discussion. The Introduction should motivate *why* a review is needed, not preview its findings. |
| S19 | Consider reporting Cohen's kappa or percent agreement for the dual-reviewer screening to substantiate the "full agreement" claim. |
| S20 | Check whether the keyword list (7 keywords) is within the target journal's limit. Some journals cap at 5–6. |

---

## 9. Final Recommendation

**Decision: Major Revision**

### Justification

The manuscript addresses a relevant and well-defined research question. The Discussion structure is its greatest asset, offering a logical and mostly insightful synthesis that progresses from findings through limitations to future directions. The articulation of the three-paradigm framework and the detection-vs-prediction gap are genuine contributions.

However, three issues prevent acceptance in the current form:

1. **PRISMA non-compliance on quality assessment.** A systematic review that claims PRISMA adherence but provides no risk-of-bias evaluation of included studies fails to meet the standard's core requirements. This is not a cosmetic issue—it means the review cannot distinguish between high-quality and low-quality evidence, undermining all interpretive claims.

2. **Reliance on a potentially predatory source.** The Siddiqui (2025) reference, published in a journal with no established credibility, is the most frequently cited source in the Discussion and single-handedly supports the paper's gap analysis and future work agenda. If this source is unreliable, the Discussion's most important claims collapse.

3. **Methodological transparency gaps.** The P4/P7 overlap, the search string's optical bias, missing stage-specific screening numbers, and the absence of a self-critical limitations section collectively weaken the review's reproducibility and credibility.

These issues are addressable. With a quality assessment table, transparent handling of the Siddiqui reference, acknowledgment of the P4/P7 overlap, a review limitations subsection, and the other high-priority suggestions, the manuscript could reach the standard expected of a Q1 systematic review. The authors are encouraged to revise thoroughly and resubmit.
