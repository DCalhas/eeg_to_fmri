"""Additional file 1 (supplementary methods) for the compact Brain Informatics manuscript: the detailed Methods of the
long version (submission/manuscript_long.tex) plus the contrastive-method material of the previous Additional file 1."""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "submission")
L = open(os.path.join(S, "manuscript_long.tex")).read()
old = open(os.path.join(S, "additional_file_1_old.tex")).read()
summ = json.load(open(os.path.join(HERE, "data", "revision_summary.json")))

def sec(start, end):
	i = L.index(start); j = L.index(end, i)
	return L[i + len(start):j].strip()

H = lambda name: r"\subsection*{" + name + "}"
datasets = sec(r"\subsection{Datasets}\label{materials:data}", r"\subsection{Simultaneous EEG-fMRI for synthesis (NODDI)}")
datasets = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", datasets, flags=re.S)
noddi = sec(r"\subsection{Simultaneous EEG-fMRI for synthesis (NODDI)}\label{section:fmri_data}", r"\subsection{EEG harmonisation and preprocessing}")
harm = sec(r"\subsection{EEG harmonisation and preprocessing}\label{section:adaptation}", r"\subsection{Problem formulation and EEG views}")
views = sec(r"\subsection{Problem formulation and EEG views}\label{section:problem_description}", r"\subsection{Synthesizers and the synthesized fMRI view}")
synth = sec(r"\subsection{Synthesizers and the synthesized fMRI view}\label{eeg_views:eeg_to_fmri}", r"\subsection{Contrastive sinusoid separation}")
# training + XP1 paragraphs go to S2 (pairing and training); the rest of the synthesizer section to S4
train = re.search(r"\\noindent\\textbf\{Synthesizer training\.\}.*?(?=\n\n)", synth, flags=re.S).group(0)
xp1 = re.search(r"\\noindent\\textbf\{Second training dataset \(XP1\)\.\}.*?(?=\n\n)", synth, flags=re.S).group(0)
synth = synth.replace(train, "").replace(xp1, "")
contr = sec(r"\subsection{Contrastive sinusoid separation}\label{methods:separation}", r"\subsection{Classification and statistical analysis}")
clf = sec(r"\subsection{Classification and statistical analysis}\label{section:setting}", r"\subsection{Supporting synthesis analyses}")
supp = sec(r"\subsection{Supporting synthesis analyses}\label{setting:fidelity}", r"\subsection{Ethics approval}")
checks = sec(r"\subsection{Recording conditions, artefacts and a below-chance estimate}\label{section:res_checks}", r"\subsection{Contrastive sinusoid separation}")
diag = checks[checks.index("The below-chance AUC of CNN-TAG in FEP2"):].strip()
checks_first = checks[:checks.index("The below-chance AUC of CNN-TAG in FEP2")].strip()

# previous Additional file 1: periodicity figure, proposition + proof, multi-class extension
per_fig = re.search(r"\\begin\{figure\}\[!htbp\].*?\\label\{fig:s1\}\s*\\end\{figure\}", old, flags=re.S).group(0)
prop = old[old.index(r"\begin{proposition}"):old.index(r"\end{proof}") + len(r"\end{proof}")]
multi = old[old.index(r"\section*{S3. Extension to more than two classes}") + len(r"\section*{S3. Extension to more than two classes}"):old.index(r"\end{document}")].strip()
per_text = old[old.index(r"\section*{S1. Periodicity of the cosine projection}") + len(r"\section*{S1. Periodicity of the cosine projection}"):old.index(r"\begin{figure}")].strip()

body = "\n\n".join([
	r"\section*{S1. Datasets and data checks}", datasets, checks_first,
	r"\section*{S2. Pairing EEG and fMRI, and synthesizer training}", noddi, train, xp1,
	r"\section*{S3. EEG harmonisation and artefact handling}", harm,
	r"\section*{S4. EEG views and synthesizers}", views, synth,
	r"\section*{S5. Contrastive sinusoid separation}", contr, H("Periodicity of the cosine projection"), per_text, per_fig,
	H("Separation under the random cosine projection"), prop, H("Extension to more than two classes"), multi,
	r"\section*{S6. Classification and statistics}", clf, H("Below-chance estimate of CNN-TAG in FEP2"), diag,
	r"\section*{S7. What the synthesizers learn}", supp,
])
# references to the main text
MAIN = {"tab:data_description": "Table~1 of the main text", "fig:overview": "Fig.~1 of the main text", "tab:auc": "Table~2 of the main text",
		"tab:sens": "Table~3 of the main text", "tab:checklist": "Table~4 of the main text", "fig:results": "Fig.~2 of the main text",
		"fig:examples": "Fig.~3 of the main text"}
SECS = {"section:adaptation": "Section~S3", "section:fmri_data": "Section~S2", "section:setting": "Section~S6", "section:validation": "Section~S6",
		"setting:fidelity": "Section~S7", "methods:separation": "Section~S5", "section:problem_description": "Section~S4",
		"eeg_views:eeg_to_fmri": "Section~S4", "materials:data": "Section~S1", "section:res_checks": "Section~S6", "section:discussion": "the Discussion of the main text"}
for lab, txt in MAIN.items():
	body = re.sub(r"(Table|Fig\.|Figure)~\\ref\{" + re.escape(lab) + r"\}", txt, body)
for lab, txt in SECS.items():
	body = re.sub(r"Section[~ ]\\ref\{" + re.escape(lab) + r"\}", txt, body)
body = body.replace("Additional file~1 gives the statement, its proof and an illustration of the periodicity problem", "the statement, its proof and an illustration of the periodicity problem are given below")
body = body.replace("(Additional file~1, Fig.~S1)", "(Fig.~\\ref{fig:s1})")
body = body.replace("Full-precision intervals, bootstrap $p$-values and Bonferroni-level intervals for all comparisons are given in Additional file~2.", "")

doc = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts,amsthm,graphicx,xcolor,booktabs,array}
\usepackage{algorithm,algorithmicx,algpseudocode}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{tikz,pgfplots,scalerel}
\usetikzlibrary{positioning, decorations.pathreplacing, calligraphy, calc, shapes.geometric, intersections, arrows}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}
\newcommand{\fmri}{functional magnetic resonance imaging}\newcommand{\FMRI}{Functional magnetic resonance imaging}
\newcommand{\fmriabbrv}{fMRI}\newcommand{\eeg}{electroencephalography}\newcommand{\EEG}{Electroencephalography}\newcommand{\eegabbrv}{EEG}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\thealgorithm}{S\arabic{algorithm}}
\newtheorem{proposition}{Proposition}
\renewcommand{\theproposition}{S\arabic{proposition}}
\title{Additional file 1: Supplementary methods\\[4pt]\large What Does EEG-to-fMRI Synthesis Add to EEG-Based Diagnosis? A Controlled Benchmark in Psychosis, Schizophrenia and Parkinson's Disease}
\author{\normalsize David Calhas and Rui Henriques}
\date{}
\begin{document}
\maketitle

""" + body + r"""

\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
"""
open(os.path.join(S, "additional_file_1.tex"), "w").write(doc)
print("AF1 words:", len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", body).split()))
