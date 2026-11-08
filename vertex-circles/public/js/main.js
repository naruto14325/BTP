// Entry point: start each section,
// and drive all animations from a single requestAnimationFrame loop.

import { initHero } from './hero.js';
import { initEyebrow } from './eyebrow.js';
import { initSpeed } from './speed.js';
import { initHip } from './hip.js';
import { initLimbs } from './limbs.js';
import presets from './presets.js';

// Only animate sections that are on screen.
function trackVisibility(sections) {
  if (!('IntersectionObserver' in window)) return;
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const section = sections.find((s) => s.el === entry.target);
      if (section) section.visible = entry.isIntersecting;
    }
  }, { rootMargin: '100px' });
  sections.forEach((s) => observer.observe(s.el));
}

// Highlight the nav link for the section in the middle of the viewport.
function highlightNav() {
  if (!('IntersectionObserver' in window)) return;
  const links = [...document.querySelectorAll('nav.side a')];
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === `#${entry.target.id}`));
    }
  }, { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('#circle, section[id]').forEach((el) => observer.observe(el));
}

function startLoop(sections) {
  let last = null;
  function frame(ts) {
    const dt = last == null ? 0 : Math.min(0.1, (ts - last) / 1000);
    last = ts;
    for (const s of sections) {
      if (s.visible) s.tick(dt);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function main() {
  initHero();
  highlightNav();
  
  const sections = [
    initEyebrow(presets),
    initSpeed(presets),
    initHip(presets),
    initLimbs(presets),
  ];
  trackVisibility(sections);
  startLoop(sections);
}

main();
