You are operating in FULL RESEARCH PROJECT REVIEW + FAIRNESS DATA ANALYSIS MODE.

Objective:
This project uses a research dataset for fairness analysis (e.g., the "FairnessForVulnerableGroups" dataset, making predictions and evaluating fairness). You also have reviewer feedback in review.txt and additional requested changes in changes.txt.

Your job is to:
1) Analyze the dataset preparation pipeline
2) Understand data preprocessing, train/test splits, handling of protected attributes
3) Evaluate model fairness logic and metrics
4) Produce a prioritized correction plan to address reviewer comments + requested changes

Critical Rules:
- DO NOT blindly rewrite the entire project.
- Respect original research goals and methodology.
- Preserve core experimental outcomes unless review requests changes.
- Improvements must be academically sound and justified.
- When adjusting dataset preparation, ensure fairness evaluation logic is correctly implemented based on accepted metrics.

-----------------------------------------
PHASE 1 — FULL PROJECT + DATASET SCAN
-----------------------------------------

1. Recursively scan the entire project directory.
2. Understand:
   • Folder and dataset structure
   • Dataset loader scripts
   • Feature engineering and protected attribute handling
   • Model training logic
   • Fairness evaluation metrics used (e.g., statistical parity, disparate impact, etc.) :contentReference[oaicite:1]{index=1}
   • Any bias mitigation strategies
3. Read:
   • review.txt (reviewer remarks)
   • changes.txt (requested modifications)
   and extract all feedback items.

Then produce:
A. Project Summary — high-level overview  
B. Dataset Pipeline Summary — how raw data moves through preprocessing  
C. Fairness Metrics Summary — what metrics are used and how  
D. Identified Risk Areas

-----------------------------------------
PHASE 2 — REVIEW & CHANGES ANALYSIS
-----------------------------------------

For each comment in review.txt and requested change in changes.txt:

1. Classify the issue type:
   - Data preparation / preprocessing
   - Fairness evaluation / bias analysis
   - Model training logic
   - Statistical validation / metrics
   - Paper writing / clarity
   - Experimental reproducibility
   - Code quality or structure
   - Ethical / protected attributes handling

2. For each issue, produce:
   - Description of problem
   - Why reviewer flagged it (data fairness imbalance? missing evaluation? unclear interpretation?)
   - Severity (Low / Medium / High / Critical)
   - Whether it requires:
        a) Code fix
        b) Dataset adjustment
        c) New evaluation / rerun experiments
        d) Paper rewrite
        e) Additional documentation

-----------------------------------------
PHASE 3 — SAFE CORRECTION PLAN
-----------------------------------------

Now produce a FIX PLAN with step-by-step actions:

1. Steps must include:
   - Files impacted
   - Dataset sections
   - Fairness measures involved
   - Risk Level
   - Reasoning for the change
   - Whether reproducibility is affected

2. Special dataset tasks:
   - Examine class imbalance issues
   - Validate correctness of protected attribute labeling
   - Check fairness metrics (e.g., statistical parity, equalized odds)
   - Adjust preprocessing only if higher fairness validity is achieved

3. Clarify any changes to metrics thresholds, altered metrics, or additional metrics added

4. Do not implement code yet — only plan

-----------------------------------------
PHASE 4 — OPTIONAL IMPROVEMENTS
-----------------------------------------

After core fix plan, list OPTIONAL improvements:
- Additional fairness evaluation techniques (e.g., disparate impact ratio checks)
- More robust protected group analysis
- Per-subgroup performance evaluation
- Additional visualizations for fairness comparisons
- Larger cross-validation evaluation (if justified)

Mark all as OPTIONAL with clear rationale.

-----------------------------------------
OUTPUT FORMAT REQUIRED
-----------------------------------------

Return output in the following structure:

1. HIGH-LEVEL PROJECT + DATASET UNDERSTANDING
2. DATA PREPARATION PIPELINE SUMMARY
3. REVIEW NOTES BREAKDOWN
4. CHANGES.TXT BREAKDOWN
5. ISSUE SEVERITY TABLE
6. PROPOSED FIX PLAN (NUMBERED STEPS)
7. OPTIONAL IMPROVEMENTS
8. CONFIRMATION CHECK REQUEST

Do not modify files before confirmation.
