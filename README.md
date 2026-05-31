# Psychological Defense Mechanism Detection with Encoder-only Architectures

**PsyDefDetect@BioNLP 2026 — 3-CFU Project Work**
Francesco Candoli · MSc in Artificial Intelligence, University of Bologna

This repository contains the report and the notebooks for a project addressing the
[PsyDefDetect shared task](https://psydefdetect-shared-task.github.io/) (BioNLP @ ACL 2026):
identifying and classifying **psychological defense mechanisms** in emotional support
dialogues, framed as a 9-class classification problem over the DMRS maturity scale.

The central question of the project is whether it is possible to capture psycholinguistic
defense patterns **without resorting to Large Language Models**, relying instead on lighter,
more sustainable **encoder-only architectures**.

## Approach

Two structurally opposed architectures were built, both on top of a pretrained encoder
(best performing: `mental/mental-bert-base-uncased`):

- **Multitask Distance Learning** — a monolithic model where concatenated seeker turns are
  encoded with MentalBERT (last two layers fine-tuned) and mean-pooled. One main 9-class head
  is supported by four auxiliary binary heads (is-it-a-defense, image-distorting, cognitive,
  mature). A 9×9 semantic-distance matrix adds a distance penalty to the loss:
  `L = L_CE + λ_d · L_dist + λ_a · Σ L_BCE`.

- **Hierarchical MIL** — the 9-way decision is decomposed across four specialized nodes
  (defense/non-defense/ambiguous → defense type → image-distorting → reality-avoidance).
  The encoder is frozen and embeddings are precomputed; each node is a Gated Attention
  Multiple Instance Learning classifier
  ([Ilse et al., 2018](https://proceedings.mlr.press/v80/ilse18a.html)) over a bag of the
  last seeker turns.

## Results (test macro-F1)

| Architecture            | Test macro-F1 |
|-------------------------|:-------------:|
| Multitask Distance      | 0.254         |
| Hierarchical MIL        | 0.259         |

Both architectures comfortably beat the frequency-proportional random baseline (≈ 0.11) and,
against the final leaderboard, would have ranked roughly 34th and 33rd out of 172 participants.
The most notable finding is that two very different designs converge on nearly identical
performance: the main limitation appears to lie in the **intrinsic separability of the data**
rather than in the architectural choices.

## Data

The task provides **PsyDefConv** (Na et al., 2025/2026), a conversational corpus annotated
with DMRS-based defense levels and derived from a stratified subset of **ESConv**
([Liu et al., 2021](https://aclanthology.org/2021.acl-long.269/)): 200 dialogues, 4,709
utterances, 2,336 annotated help-seeker turns. A balanced validation set (15 samples per class)
was carved out, and the training set was extended with prompt-based synthetic dialogues.
Back-translation and TF-IDF clustering reduction were also tried but discarded after they
degraded results.

## Repository structure

| File | Description |
|------|-------------|
| `data_processing_NLP.ipynb` | Split creation, synthetic-data augmentation, and per-node subset preparation. |
| `multitask_learning_NLP.ipynb` | Multitask Distance Learning architecture (training, evaluation, distance/auxiliary losses). |
| `hierarchical_mil_NLP.ipynb` | Hierarchical Gated-Attention MIL architecture (per-node training and full pipeline). |
| `report.pdf` | Project report. |

## References

- Na et al. (2026). *Overview of the PsyDefDetect Shared Task at BioNLP 2026.*
- Na et al. (2025). *You never know a person, you only know their defenses* (PsyDefConv dataset).
- Liu et al. (2021). *Towards Emotional Support Dialog Systems* (ESConv).
- Ji et al. (2022). *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.*
- Ilse, Tomczak, Welling (2018). *Attention-based Deep Multiple Instance Learning.*
- Di Giuseppe & Perry (2021). *The Hierarchy of Defense Mechanisms (DMRS Q-sort).*
