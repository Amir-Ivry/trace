"use strict";
// Native controls remain usable without JavaScript; never autoplay.
const players = [...document.querySelectorAll("audio")];
players.forEach(player => player.addEventListener("play", () => {
  players.forEach(other => { if (other !== player) other.pause(); });
}));

const copyButton = document.getElementById("copy-citation");
copyButton.addEventListener("click", async () => {
  const citation = document.getElementById("bibtex");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = "Citation copied.";
  } catch (_) {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Use your browser’s Copy command.";
  }
});
