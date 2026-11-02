\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{imag-ms-template}[2023/10/10 v1.3
  LaTeX class file for MIT Journals applicable for preparing manuscript at author's end]

\LoadClass[twoside,12pt]{article}

\usepackage{mathtools,amsthm,amsfonts,amssymb}%
\usepackage[doublespacing]{setspace}%
\usepackage{threeparttable,booktabs}%
\usepackage{graphicx}%
\usepackage[figuresright]{rotating}%
\usepackage{enumitem}%

\newcommand{\keywords}[1]{\bgroup\noindent\textbf{Keywords:}
#1\egroup}%

\renewenvironment{abstract}{%
\begin{quote}\textbf{Abstract:} }%
{\end{quote}}%

\usepackage[natbib=true,backend=biber,style=apa]{biblatex}

\usepackage[ruled]{algorithm2e}%

\usepackage{mathrsfs}%

\newtheorem{theorem}{Theorem}%
\newtheorem{lemma}{Lemma}%
\newtheorem{corollary}{Corollary}%
\newtheorem{remark}{Remark}%

\newtheorem*{theorem*}{Theorem}%
\newtheorem*{lemma*}{Lemma}%
\newtheorem*{corollary*}{Corollary}%
\newtheorem*{remark*}{Remark}%

\usepackage[breaklinks=true]{hyperref}%

\AtBeginDocument{\pagestyle{headings}}%

\newcommand{\notetocomp}[1]{\bgroup\parindent\z@\raggedright%
\marginpar{\vbox{\hsize=6pc\footnotesize\raggedright#1}\par}
\egroup}%

%%%%%%%%%%%%%% additions by Steve Smith 11/1/24

%%% margins
%\topmargin -1.5 cm
%\textheight 23.5 cm
%\oddsidemargin -1.3 cm
%\evensidemargin -1.3 cm
%\textwidth 18 cm

\topmargin -1.2 cm
\textheight 23.1 cm
\oddsidemargin -0.6 cm
\evensidemargin -0.6 cm
\textwidth 17.8 cm

%%% paragraph indentation and line spacing
\parindent 0cm
\parskip 0.3cm
\linespread{1.3}

%%% uncomment the following lines to include line numbers
%\usepackage{lineno}
%\linenumbers

%%% switch to sans serif font
\renewcommand{\familydefault}{\sfdefault}

%%%%%%%%%%%%%% end additions by Steve Smith 11/1/24

\endinput
