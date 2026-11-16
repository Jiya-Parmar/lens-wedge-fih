# AI-use disclosure

AI tool used: Claude (Anthropic), as permitted by the ES 678 project rules.

What it was used for:
* writing and debugging Python code in `src/` and `scripts/` (data loading, features, models,
  evaluation, the FIH representation, diagnostics and figures);
* drafting and editing the README, HOW_TO_RUN and the LaTeX paper;
* checking that results reproduce across Python and library versions.

What I checked myself:
* every number reported in the paper against the printed script outputs and the CSV files in
  `results/`;
* that a full rerun with `bash run_all.sh` reproduces the reported numbers;
* the invariance proof in the paper, and its numerical check (script 09, part C).

I am responsible for all code, results and text, and can explain any part of them.

External material: the robot pipeline illustration in Figure 1 of the paper comes from the
internship project in which the data were collected. Methods are cited in the paper.
