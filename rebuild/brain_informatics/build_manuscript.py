"""Convert rebuild/paper_ni/main.tex (article class, inline bibliography) into a Brain Informatics submission
(Springer Nature sn-jnl template, Basic Springer reference style, double spacing + line numbers, Declarations).
Re-run after editing paper_ni/main.tex; the BI-specific blocks (abstract, keywords, abbreviations, declarations,
ethics and LLM statements) are defined here."""
import re, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "paper_ni", "main.tex")
TPL = os.path.join(HERE, "template", "sn-article-template")
OUT = os.path.join(HERE, "submission")
os.makedirs(OUT, exist_ok=True)
for f in ["sn-jnl.cls", "bst/sn-basic.bst"]:
	shutil.copy(os.path.join(TPL, f), OUT)
for f in ["stft.png", "fmri.png"]:
	shutil.copy(os.path.join(HERE, "..", "paper_ni", f), OUT)

s = open(SRC).read()
body = s[s.index("\\maketitle") + len("\\maketitle"):s.index("\\bibliographystyle{unsrtnat}")]
appendix = s[s.index("\\begin{proof}"):s.index("\\end{proof}") + len("\\end{proof}")]

# drop the article-class abstract, the old Limitations/Social impact/Ethics tail (re-added below in BI form)
body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.S)
lim = re.search(r"\\section\*\{Limitations\}(.*?)\\section\*\{Social impact\}", body, flags=re.S).group(1).strip()
body = body[:body.index("\\section*{Limitations}")]
# Limitations become a subsection of the Discussion (placed before Related work)
body = body.replace("\\section{Related work}", "\\subsection{Limitations}\\label{section:limitations}\n\n" + lim + "\n\n\\section{Related work}")
# appendix reference for the proof
body = body.replace("(proof in the Appendix)", "(proof in Appendix~\\ref{app:proof})")

ETHICS_LLM = r"""
\subsection{Ethics approval and use of language models}\label{section:ethics}

This study is a secondary analysis of anonymised, publicly released data; no new data were collected. All original studies were approved by the responsible committees and obtained informed consent from participants, as reported by the data providers: NODDI, approval ID 4290/001 \cite{dataset_01}; first-episode psychosis datasets, University of Pittsburgh \verify{approval ID} \cite{salisbury_ds003944,salisbury_ds003947}; Semmelweis dataset, Semmelweis University \verify{approval ID} \cite{racz2025}; Parkinson's disease dataset, University of New Mexico \verify{approval ID} \cite{cavanagh_ds003490}; Fribourg dataset, approval ID 054/13-CER-FR \cite{dataset2022fribourg}. The analyses were carried out in accordance with the relevant guidelines and regulations and the Declaration of Helsinki.

A large language model (Claude, Anthropic) was used to assist with writing analysis code and drafting and editing the manuscript text. All analyses, results and text were reviewed and verified by the authors, who take full responsibility for the content. \verify{adjust to the actual use}
"""
body = body.replace("\\subsection{Dataset Characteristics}\\label{materials:data}", ETHICS_LLM + "\n\\subsection{Dataset Characteristics}\\label{materials:data}")

# article-class styling that sn-jnl handles differently
body = body.replace("\\begin{table*}[t]", "\\begin{table*}[!t]")
# sn-jnl's text block is narrower than article's (~10.5 cm of p{} width for six columns): explicit column widths
COLS = {
	"{p{1.9cm} | p{2.2cm} | p{0.9cm} | p{2.4cm} | p{1.6cm} | p{3.6cm}}": "{p{1.75cm} | p{1.95cm} | p{0.85cm} | p{2.0cm} | p{1.3cm} | p{2.65cm}}",
	"{ p{1.8cm} | p{3.2cm} | p{1.8cm} | p{1.2cm} | p{3.0cm} | p{1.6cm}}": "{@{}l l c c c c@{}}",
	"{ p{1.6cm} | p{3.4cm} | p{3.4cm} | p{1.4cm}}": "{@{}l c c c@{}}",
}
for k, v in COLS.items():
	assert k in body, k
	body = body.replace(k, v)
# result tables (natural-width columns) in a smaller font; let floats be placed here or on float pages
for lab in ["tab:clf", "tab:contrastive"]:
	i = body.index("\\label{" + lab + "}")
	j = body.index("\\small", i)
	assert j - i < 40, lab
	body = body[:j] + "\\footnotesize\\setlength{\\tabcolsep}{4pt}" + body[j + len("\\small"):]
body = re.sub(r"\\begin\{table\}\[(t|ht|h)\]", r"\\begin{table}[!htbp]", body)

# figure titles (Brain Informatics: title of at most 15 words, then the legend)
FIG_TITLES = {
	"fig:horizontal_architecture": "Architecture of the CNN-TAG synthesizer",
	"fig:normalization_shift": "Periodicity of the cosine projection",
	"fig:sinusoid_separation": "Contrastive separation of sinusoids",
	"future_work:fig:multi_label_sinusoid": "Sinusoid separation for more than two classes",
}
FIGS = []  # (number, label, tikz source) for the separate figure files
for m in re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", body, flags=re.S):
	fig = m.group(0)
	lab = re.search(r"\\label\{([^}]*)\}", fig)
	lab = lab.group(1)
	FIGS.append((len(FIGS) + 1, lab, re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", fig, flags=re.S).group(0)))
for _, lab, _ in FIGS:
	assert lab in FIG_TITLES, lab
def _title(m):
	fig = m.group(0)
	lab = re.search(r"\\label\{([^}]*)\}", fig)
	lab = lab.group(1)
	return fig.replace("\\caption{", "\\caption{\\textbf{" + FIG_TITLES[lab] + "} ", 1)
body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", _title, body, flags=re.S)

PREAMBLE = r"""\documentclass[pdflatex,referee,sn-basic]{sn-jnl}% Basic Springer Nature reference style; referee = double spacing
% Line numbers on the text lines (lineno package). The class's own `lineno' option draws a fixed margin ruler
% at single-spacing pitch, which with double spacing gives two numbers per line and numbers beside blank space.
\usepackage[left,mathlines]{lineno}

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{booktabs}%
\usepackage{algorithm}%
\usepackage{algorithmicx}%
\usepackage{algpseudocode}%
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{scalerel}
\usetikzlibrary{positioning, decorations.pathreplacing, calligraphy, calc, shapes.geometric, intersections, arrows}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}

\newcommand{\fmri}{functional magnetic resonance imaging}
\newcommand{\FMRI}{Functional magnetic resonance imaging}
\newcommand{\fmriabbrv}{fMRI}
\newcommand{\eeg}{electroencephalography}
\newcommand{\EEG}{Electroencephalography}
\newcommand{\eegabbrv}{EEG}
\newcommand{\verify}[1]{\textcolor{red}{[VERIFY: #1]}}

\theoremstyle{thmstyleone}%
\newtheorem{theorem}{Theorem}%

\raggedbottom

\begin{document}
\linenumbers

\title[Does EEG to fMRI Synthesis Aid Medical Decision?]{Does EEG to fMRI Synthesis Aid Medical Decision? An Evaluation in Psychosis, Schizophrenia and Parkinson's Disease}

\author*[1]{\fnm{David} \sur{Calhas}}\email{davidcalhas@gmail.com}
\author[2]{\fnm{Rui} \sur{Henriques}}\email{\verify{email}}

\affil*[1]{\orgdiv{\verify{department}}, \orgname{\verify{institution}}, \orgaddress{\city{\verify{city}}, \country{\verify{country}}}}
\affil[2]{\orgdiv{\verify{department}}, \orgname{\verify{institution}}, \orgaddress{\city{\verify{city}}, \country{\verify{country}}}}

\abstract{Functional magnetic resonance imaging (fMRI) maps brain activity in space but is expensive, while electroencephalography (EEG) is cheap and portable. EEG-to-fMRI synthesis has been proposed to bring fMRI-like information to EEG-only clinical settings; we tested whether it improves diagnosis. Four synthesizers were trained on a simultaneous EEG-fMRI dataset of healthy adults with scanner-trigger alignment and applied to resting EEG from five public cohorts: two first-episode psychosis samples (77 and 61 participants), chronic schizophrenia (77), Parkinson's disease (50) and a task-EEG schizophrenia-spectrum sample (43). Each cohort was classified from EEG band power, synthesized fMRI and their combination with linear models under repeated cross-validation and permutation tests, with untrained (random-weight) synthesizers as an architecture-only control and contrastive fine-tuning of the synthesizers evaluated alongside. Band power separated Parkinson's disease from controls (area under the ROC curve, AUC, 0.79) but not the psychosis or schizophrenia cohorts (0.54--0.65). Synthesized fMRI classified Parkinson's disease (best AUC 0.74) but stayed below band power there, and elsewhere only one synthesized view was significant, with its untrained version scoring higher. Adding synthesized fMRI to band power gave no gain that survived correction for multiple comparisons, untrained synthesizers reproduced the Parkinson's disease result (AUC 0.70--0.74), and contrastive fine-tuning reached at most 0.70. In supporting analyses, the synthesizers did not outperform an EEG-free template image, and EEG explained a negligible part of preprocessed blood-oxygen-level-dependent signal (correlation about 0.03). Synthesized fMRI carried EEG spectral information rather than learned haemodynamics and did not reliably improve EEG-based diagnosis in these cohorts.}

\keywords{Electroencephalography, EEG-to-fMRI synthesis, Functional MRI, Schizophrenia, First-episode psychosis, Parkinson's disease, Diagnostic classification, Cross-modal generation, Benchmark}

\maketitle
"""

ABBREV = r"""
\section*{List of abbreviations}
AUC: area under the receiver operating characteristic curve; BOLD: blood-oxygen-level-dependent; BPRS: Brief Psychiatric Rating Scale; CI: confidence interval; CNN-TAG: convolutional neural network with topographical attention graph; ECG: electrocardiogram; EEG: electroencephalography; EOG: electrooculogram; EPI: echo-planar imaging; FEP: first-episode psychosis; fMRI: functional magnetic resonance imaging; IAAFT: iterative amplitude-adjusted Fourier transform; ICD-10: International Classification of Diseases, 10th revision; LOOCV: leave-one-out cross-validation; MSE: mean squared error; PANSS: Positive and Negative Syndrome Scale; PCA: principal component analysis; PD: Parkinson's disease; PSNR: peak signal-to-noise ratio; RMSE: root mean squared error; SSIM: structural similarity index measure; STFT: short-time Fourier transform; TR: repetition time.
"""

BACK = r"""
\backmatter

\section*{Declarations}

\bmhead{Availability of data and materials}
All datasets analysed in this study are publicly available: NODDI simultaneous EEG-fMRI (Open Science Framework, \url{https://osf.io/94c5t}) \cite{dataset_01}; first-episode psychosis resting EEG, OpenNeuro ds003944 and ds003947 \cite{salisbury_ds003944,salisbury_ds003947}; Semmelweis schizophrenia resting EEG, Zenodo record 14808296 \cite{racz2025}; Parkinson's disease resting EEG, OpenNeuro ds003490 \cite{cavanagh_ds003490}; Fribourg Ultimatum Game EEG, OpenNeuro ds004000 \cite{dataset2022fribourg}. The code is available at \verify{repository URL and archived version (e.g. Zenodo DOI)}. Project name: \verify{name}; operating system: Linux; programming language: Python; other requirements: PyTorch, MNE-Python, scikit-learn, FSL; licence: \verify{licence}.

\bmhead{Competing interests}
The authors declare that they have no competing interests. \verify{confirm}

\bmhead{Funding}
\verify{funding sources and grant numbers, or `Not applicable'}

\bmhead{Authors' contributions}
\verify{e.g. DC designed the study, implemented the methods, performed the analyses and wrote the manuscript. RH supervised the study and revised the manuscript. All authors read and approved the final manuscript.}

\bmhead{Acknowledgements}
We thank the groups that collected and publicly released the datasets used in this study. \verify{add any further acknowledgements}

\begin{appendices}

\section{Proof of Theorem 1}\label{app:proof}

""" + appendix.replace("\\begin{proof}", "\\begin{proof}[Proof of Theorem 1]") + r"""

\end{appendices}

\bibliography{references}

\end{document}
"""

out = PREAMBLE + body.rstrip() + "\n" + ABBREV + BACK
open(os.path.join(OUT, "manuscript.tex"), "w").write(out)

# bibliography: original 2025 references.bib for cited keys + new entries
cites = set(c.strip() for g in re.findall(r"\\cite[pt]?\{([^}]*)\}", out) for c in g.split(","))
def entries(path):
	txt = open(path).read()
	chunks = re.split(r"\n(?=@)", "\n" + txt)
	return {re.match(r"@\w+\{([^,]+),", c.strip()).group(1).strip(): c.strip() for c in chunks if c.strip().startswith("@")}
orig = entries(os.path.join(HERE, "..", "paper_ni", "references.bib"))
new = entries(os.path.join(HERE, "..", "paper", "references.bib"))
if "jiang2022labram" in new:
	new["jiang2024labram"] = new.pop("jiang2022labram").replace("{jiang2022labram,", "{jiang2024labram,")
bib, missing = [], []
for k in sorted(cites):
	e = orig.get(k) or new.get(k)
	(bib.append(e) if e else missing.append(k))
bib = [re.sub(r"doi\s*=\s*\{doi:", "doi = {", e) for e in bib]
open(os.path.join(OUT, "references.bib"), "w").write("% Entries with note={VERIFY...} must be checked against the source.\n\n" + "\n\n".join(bib) + "\n")
words = len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", out[out.index("\\abstract{") + 10:out.index("\\keywords")]).split())
print("written", OUT, "| missing bib:", missing, "| abstract words ~", words)

# figures as separate files (standalone PDFs at the manuscript's text width)
FIG_PRE = PREAMBLE[PREAMBLE.index("\\usepackage{tikz}"):PREAMBLE.index("\\theoremstyle")]
for n, lab, tikz in FIGS:
	open(os.path.join(OUT, f"fig{n}.tex"), "w").write(
		"\\documentclass[border=4pt]{standalone}\n\\usepackage{graphicx,amsmath,amssymb,amsfonts,xcolor}\n" + FIG_PRE
		+ "\\setlength{\\textwidth}{372pt}\n\\begin{document}\n" + tikz + "\n\\end{document}\n")
print("figures:", [(n, lab) for n, lab, _ in FIGS])
