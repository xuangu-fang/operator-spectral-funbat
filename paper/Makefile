# Operator-Spectral FunBaT — paper build
LATEXMK ?= latexmk

.PHONY: all watch clean distclean report-zh

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode main.tex

# Render the Chinese technical report to PDF for local reading / formula checking.
# Requires pandoc + xelatex + fonts-noto-cjk.
report-zh:
	pandoc ../docs/PAPER_TECHNICAL_REPORT_ZH.md \
	  -o ../docs/PAPER_TECHNICAL_REPORT_ZH.pdf \
	  --pdf-engine=xelatex \
	  -V CJKmainfont="Noto Serif CJK SC" \
	  -V mainfont="DejaVu Serif" \
	  -V geometry:margin=2.2cm \
	  --toc --resource-path=..:../docs

clean:
	$(LATEXMK) -c
	rm -f *.bbl *.run.xml

distclean:
	$(LATEXMK) -C
	rm -f *.bbl *.run.xml
