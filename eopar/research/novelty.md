# Research hypotheses — not established novelty

## Problem
Attribute existence and whether an image permits observing that attribute are
different targets. A shoe can exist while its evidence is occluded. Ordinary
classification or consistency alone cannot validate whether a model recognizes
that distinction. Synthetic body bands provide interventions, not natural
observability ground truth. Image quality, viewpoint and multiple evidence
locations can invalidate those bands.

## Hypotheses
1. Relative observability ranking after evidence removal improves observability
   discrimination compared with the old visibility head and prediction entropy.
2. Restricting invariance to intact evidence preserves classification robustness
   without forcing confident classification on a removed evidence region.
3. Improvements transfer to unseen occluders, natural occlusion and held-out
   domains without a material clean-performance regression.

## Closest prior work

| Work | Established overlap | Proposed distinction / evidence still required |
|---|---|---|
| [UAPAR](https://arxiv.org/html/2604.26873v1), method full text inspected | CLIP fusion, region priors, Beta evidence and uncertainty-guided curriculum | Separate supervised observability plus intervention ranking; compare to its uncertainty and rejection curves, not only mA |
| [AAR](https://doi.org/10.1016/j.neucom.2025.130236), abstract inspected | Perturbations supervise attribute attention | Target observability change rather than attention quality; full loss comparison remains OPEN |
| [MambaPAR](https://arxiv.org/abs/2407.10374), abstract inspected | Mamba and hybrid architectures | No novelty claimed for the backbone combination |
| [UniPAR](https://doi.org/10.1049/cvi2.70081), abstract inspected | Attribute queries and semantic fusion | No novelty claimed for queries/fusion |
| [MSP60K](https://ojs.aaai.org/index.php/AAAI/article/view/32434), abstract inspected | Partial-aware queries and cross-domain evaluation | Existing dataset/splits reused; do not claim a new image dataset |

UAPAR models predictive uncertainty through Beta evidence and uses region priors
and a training curriculum. EOPAR instead tests a separate observability output
against controlled evidence removal. This is a proposed distinction, not proof
that no related method exists. It must outperform max binary probability and
entropy on identical retained decisions. Observability is not automatically
epistemic uncertainty, calibration, or a causal effect.

## Gate
Phase 0 remains OPEN: 26 paper candidates are registered, but fewer than 15 full
texts have been critically reviewed. Full AAR comparison, citation snowballing,
and independent dataset/code checks are required. No "first", "SOTA", or Q1
readiness claim is authorized by these notes. Short integration pilots may run;
large model selection runs wait for the literature/data gates.

## Journal shortlist
Pattern Recognition, TCSVT, Neural Networks are candidates by topic. JCR JIF
quartile/category/year is UNVERIFIED. Record institutional JCR evidence and the
school's accepted edition before choosing. SJR and impact factor alone do not
establish JCR Q1: https://clarivate.libguides.com/jcr

Direct baseline provenance: [CLIMP](https://arxiv.org/abs/2601.06891) is explicitly referenced by the existing encoder source. Cite that inspiration and distinguish independently pretrained towers from jointly contrastive-pretrained CLIMP weights. No claim is made that existing MSP models loaded official CLIMP joint weights.
