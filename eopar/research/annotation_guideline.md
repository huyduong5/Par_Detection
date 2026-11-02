# Natural observability annotation protocol

Target: 800 human-screened naturally occluded images, 200 from validation and
600 from test. Candidate sampling is not confirmation of natural occlusion.
Use the existing partitions, never training images. First screen candidates;
stratify accepted images by head/upper/lower/shoes and mild/moderate/severe
occlusion. Log selection counts and rejection reasons. Do not cherry-pick on
model errors or predictions. Check repeated identities if metadata is available.

Annotators see image, attribute definition and observation rules; they do NOT see
attribute ground-truth labels, model predictions, or the other annotator's work.
The task is whether available evidence supports deciding the attribute either
present OR absent, not whether the attribute is positive.

- visible: relevant evidence allows a defensible present/absent judgment.
- partial: relevant evidence is partly available, but cannot support the same
  unqualified judgment as visible.
- unobservable: relevant evidence is absent or unusable; do not infer from
  demographic associations, other people, or background.
- uncertain: attribute definition or evidence is ambiguous; request adjudication.

For example, clearly seeing the shoe region can justify absence of boots; a hidden
shoe region cannot. Original images are not automatically visible. Global or
subjective attributes need a separately agreed rubric; the initial release
excludes them from observability evaluation.

Two annotators independently label at least 25% of images (50 validation + 150
test). Report agreement, confusion and Cohen kappa, then have a third reviewer
adjudicate disagreements. Preserve both original annotations and adjudications.
Uncertain/partial decisions are counted; binary AUC uses visible vs unobservable
only. Never silently drop these decisions from standard PAR metrics.

Use `python -m eopar.scripts.annotations --candidates candidates.csv --output
annotator_a.csv`; input columns are image,split,attribute. Copy the blank sheet
for annotator B. Both may include all images, but at least the specified overlap
must be labeled twice. Use --first / --second / --output for the agreement report.
Do not mark unfilled annotation sheets as completed datasets.

Freeze final test rows and their SHA256 before model selection. Validation alone
sets thresholds. Keep annotator identifiers pseudonymous. Store image references
and annotations according to source dataset licensing; do not redistribute images
without permission.
