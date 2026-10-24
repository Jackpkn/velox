// Full-page text with link targets, for read_page and extract. It reads the page's main content and
// drops chrome a reader skips: navigation, sidebars, site header/footer, cookie banners, empty bullets.
// Links are compacted: tracking redirects are unwrapped, tracking/opaque query params removed, and a
// target already printed once is not printed again.
(budget) => {
  const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'NAV', 'ASIDE', 'IFRAME']);
  const SKIP_ROLES = new Set(['navigation', 'search', 'banner', 'contentinfo', 'complementary', 'menu', 'menubar']);
  const NOISE = /(^|[\s_-])(cookie|consent|gdpr|skip-?link|skip-?to)([\s_-]|$)/i;
  const TRACKING = /^(utm_|ref_?$|gclid$|fbclid$|msclkid$|mc_|_ga$|spm$|trk|tracking|sp_csd$|dib|qid$|sr$|crid$|sprefix$|pd_rd_|pf_rd_|content-id$)/i;
  const REDIRECT = ['url', 'u', 'dest', 'destination', 'redirect', 'redirect_url', 'target', 'to'];
  const here = location.href.split('#')[0];
  const seen = new Set();
  const ITEM = '\u241E';  // marks an item boundary until the text is joined

  const compactHref = (raw) => {
    let url;
    try { url = new URL(raw); } catch { return raw; }
    for (const key of REDIRECT) {
      const inner = url.searchParams.get(key);
      if (inner && /^(\/[^/]|https?:)/.test(inner)) {
        try { url = new URL(inner, url.origin); break; } catch { /* keep the outer link */ }
      }
    }
    for (const [key, value] of [...url.searchParams]) {
      if (TRACKING.test(key) || value.length > 60) url.searchParams.delete(key);
    }
    // Path segments like /ref=sr_1_7 are tracking too.
    url.pathname = url.pathname.replace(/\/ref=[^/]*$/, '');
    url.hash = '';
    return url.href;
  };

  const skipped = (n) => {
    const tag = n.tagName.toUpperCase();
    if (SKIP_TAGS.has(tag)) return true;
    if (SKIP_ROLES.has(n.getAttribute('role'))) return true;
    if (n.getAttribute('aria-hidden') === 'true') return true;
    if ((tag === 'HEADER' || tag === 'FOOTER') && !n.closest('article, main, [role=main]')) return true;
    const cls = typeof n.className === 'string' ? n.className : '';
    if (NOISE.test(n.id || '') || NOISE.test(cls)) return true;
    return n.checkVisibility && !n.checkVisibility({ checkVisibilityCSS: true });
  };

  const lines = [];
  const walk = (el) => {
    for (const n of el.childNodes) {
      if (n.nodeType === 3) { lines.push(n.textContent); continue; }
      if (n.nodeType !== 1 || skipped(n)) continue;
      if (n.tagName === 'A' && n.href && /^https?:/.test(n.href)) {
        const t = n.innerText.replace(/\s+/g, ' ').trim();
        if (!t) continue;
        const href = n.href.split('#')[0] === here ? '' : compactHref(n.href);
        if (!href || seen.has(href)) { lines.push(` ${t} `); continue; }
        seen.add(href);
        lines.push(` [${t}](${href}) `);
        continue;
      }
      const block = /^(P|DIV|LI|TR|H[1-6]|SECTION|ARTICLE|TABLE|UL|OL|BR|HEADER|FOOTER|DT|DD)$/.test(n.tagName);
      if (/^H[1-6]$/.test(n.tagName)) lines.push('\n' + '#'.repeat(+n.tagName[1]) + ' ');
      else if (block) lines.push('\n');
      const cls = typeof n.className === 'string' ? n.className : '';
      const rating = cls.split(/\s+/).some((c) => /rating/i.test(c) || /^stars?$/i.test(c));
      if (rating && !n.innerText.trim()) lines.push(` [${cls.trim().slice(0, 60)}] `);
      if (n.tagName === 'TD' || n.tagName === 'TH') lines.push(' | ');
      // A blank line between repeated items keeps each item's fields (e.g. a rating printed before the
      // title) attached to the right item.
      if (n.tagName === 'LI' || n.tagName === 'ARTICLE') lines.push(`\n${ITEM}\n`);
      if (n.tagName === 'LI') lines.push('- ');
      walk(n);
      if (block) lines.push('\n');
    }
  };

  const main = document.querySelector('main, [role=main]');
  walk(main && main.innerText.trim().length >= 500 ? main : document.body);
  const text = lines.join('')
    .replace(/[ \t]+/g, ' ')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !/^([-|#]\s*)+$/.test(l))
    .join('\n')
    .replace(new RegExp(`(\\n?${ITEM}\\n?)+`, 'g'), '\n\n')
    .trim();
  return { text: text.slice(0, budget), truncated: text.length > budget, length: text.length };
}
