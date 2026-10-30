window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// MathJax typesets the first page itself during startup. Clearing the output
// cache at that point drops the CHTML glyph styles and leaves every formula
// blank, so only re-typeset after Material's instant navigation swaps the page.
// On those swaps the CHTML font cache must be cleared too, otherwise glyphs not
// seen on an earlier page get no CSS and render as blanks.
let mathjaxFirstPage = true;
document$.subscribe(() => {
  if (mathjaxFirstPage) {
    mathjaxFirstPage = false;
    return;
  }
  if (!window.MathJax || !MathJax.typesetPromise) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
