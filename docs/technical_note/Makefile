# 用 xelatex 编译（中文需要 ctex 与 Noto CJK 字体）
main.pdf: main.tex
	xelatex -interaction=nonstopmode main.tex >/dev/null
	xelatex -interaction=nonstopmode main.tex >/dev/null
	cp main.pdf ../技术札记.pdf

clean:
	rm -f main.aux main.log main.out main.toc main.pdf

.PHONY: clean
