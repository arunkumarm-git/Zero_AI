\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vitmscthesis}[2022/08/05 MSc Thesis Template for VIT, Chennai]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass{book}

%%
\RequirePackage{graphicx, amsmath, times, enumitem, booktabs, amssymb,amsthm,longtable}
\RequirePackage{tikz}
\RequirePackage[table,dvipsnames]{xcolor}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
%%
\RequirePackage{makeidx}
\makeindex

\RequirePackage{natbib}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}

%%%--- Custom commands for use case
\newcommand{\thesistitle}[1]%
{\def\printtitle{#1} \title{#1}}%

\newcommand{\thesison}[1]{
	\def\thethesison{\MakeUppercase{#1}}
}

\newcommand{\programme}[1]{
\def\theprogramme{#1}}

\newcommand{\programmeLongName}[1]{
	\def\theprogrammeLongName{#1}}

\newcommand{\department}[1]{
	\def\thedepartment{#1}}

\newcommand{\school}[1]{
	\def\theschool{#1}}

\newcommand{\student}[1]%
{\def\thestudent{#1}\author{#1}}%

\newcommand{\regno}[1]%
{\def\theregno{#1}}%

\newcommand{\guide}[1]%
{\def\theguide{#1}}%

\newcommand{\guidedesignation}[1]%
{\def\theguidedesignation{#1}}%

\newcommand{\hod}[1]%
{\def\thehod{#1}}%

\newcommand{\submittedon}[1]{
	\def\thesubmittedon{#1}
}
\newcommand{\bibliographyname}[1]{
	\def\thebibliographyname{#1}
}

\newenvironment{keywords}{\noindent\textbf{Keywords:}}{}



%%%%%%%------------------------------------------------
%%%%%%-----------Page Layout Setups------
%%%%%%--------------------------------------------------
\RequirePackage{geometry}
\geometry{
	headheight=6ex,
	includehead,
	includefoot
}

\geometry{
	paper=a4paper,% Change to letterpaper for US letter
	left=3.81cm, % left margin
	right=2.54cm, % Outer margin
	bindingoffset=0cm, % Binding offset
	top=2.54cm, % Top margin
	bottom=2.54cm, % Bottom margin
	%showframe,% show how the type block is set on the page
}

\RequirePackage{fancyhdr}%for the header and footer
\RequirePackage{emptypage}%to remove headers and footers in the blank pages	

\pagestyle{fancy} %Initiates the Page Style

\fancyhf{}
\renewcommand{\headrulewidth}{1pt}
%\fancyhead[L]{\nouppercase \printtitle}
\fancyhead[R]{\nouppercase \leftmark}

\renewcommand{\footrulewidth}{0.5pt}
\fancyfoot[R]{\thepage}
\fancyfoot[L]{}



%%%%%------------------------------------------------
%%%%%-----------Hyperlinks Setup---------------------
%%%%%------------------------------------------------
\RequirePackage{nameref}
\RequirePackage[pdfusetitle]{hyperref}
\hypersetup{
	colorlinks=true,
	linkcolor=black,
	filecolor=red,      
	urlcolor=black,
	citecolor = black
}

\RequirePackage[french,latin,english]{babel}
\RequirePackage[utf8]{inputenc}
\addto\extrasenglish{%
	\renewcommand{\chapterautorefname}{Chapter}%
}
\addto\extrasenglish{%
	\renewcommand{\sectionautorefname}{Section}%
}



%----------------------------------------------------------------------------------------
%	PENALTIES( for the Hypenation Problem)
%----------------------------------------------------------------------------------------

\doublehyphendemerits=10000 % No consecutive line hyphens
\brokenpenalty=10000 % No broken words across columns/pages
\widowpenalty=9999 % Almost no widows at bottom of page
\clubpenalty=9999 % Almost no orphans at top of page
\interfootnotelinepenalty=9999 % Almost never break footnotes


%%%--Bibliography Related----

\AtBeginDocument{
% \bibliographystyle{agsm}
\renewcommand{\bibname}{\thebibliographyname}
}