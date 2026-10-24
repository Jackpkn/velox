// Serialize the current page for the model: visible text in reading order with interactive
// elements inlined as [id]<tag ...>label</tag>. Element ids are valid for this observation only.
(budget) => {
  const state = (window.__ha = { nodes: new Map(), next: 1 });
  const autocompleteIds = (window.__ha_autocomplete ||= {
    nodes: new WeakMap(), next: 1, document: String(performance.timeOrigin),
  });
  const INTERACTIVE_ROLES = new Set([
    'button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem', 'menuitemcheckbox',
    'menuitemradio', 'option', 'combobox', 'textbox', 'searchbox', 'spinbutton', 'slider', 'treeitem',
    'gridcell',
  ]);
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'PATH', 'HEAD', 'META', 'LINK']);
  const BLOCK = /^(ADDRESS|ARTICLE|ASIDE|BLOCKQUOTE|DD|DIV|DL|DT|FIELDSET|FIGCAPTION|FIGURE|FOOTER|FORM|H[1-6]|HEADER|HR|LI|MAIN|NAV|OL|P|PRE|SECTION|TABLE|TR|UL|BR|TD|TH|LABEL|DETAILS|SUMMARY|DIALOG)$/;
  const NESTED_CONTROL = 'input:not([type="hidden"]),select,textarea,button,a[href],[role="button"],[role="combobox"],[role="textbox"],[role="searchbox"],[role="checkbox"],[role="radio"],[role="option"],[role="tab"]';
  const vw = innerWidth, vh = innerHeight;
  const clean = (s, n = 120) => (s || '').replace(/\s+/g, ' ').trim().slice(0, n);

  const visible = (el) => !el.checkVisibility || el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
  const hasBox = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const pointer = (el) => !!el && getComputedStyle(el).cursor === 'pointer';
  // Script-driven controls (an <a> without href, a clickable <div>) are found by their pointer cursor; a
  // child that only inherits the cursor from a clickable parent is not a second target.
  const scriptClickable = (el) => pointer(el) && !pointer(el.parentElement);
  const isInteractive = (el) => {
    const tag = el.tagName;
    // A visible link with a real box is meant to be clicked even without href: SPAs attach handlers in
    // script with no attribute or pointer cursor to show for it (empty named anchors have no box).
    if (tag === 'A') return true;
    if (tag === 'INPUT') return el.type !== 'hidden';
    if (['BUTTON', 'SELECT', 'TEXTAREA', 'SUMMARY'].includes(tag)) return true;
    if (el.isContentEditable && !el.parentElement?.isContentEditable) return true;
    const role = el.getAttribute('role');
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.hasAttribute('onclick')) return true;
    const ti = el.getAttribute('tabindex');
    if (ti !== null && ti !== '-1' && pointer(el)) return true;
    return ['DIV', 'SPAN', 'IMG', 'LI', 'I'].includes(tag) && scriptClickable(el);
  };
  const accessibleName = (el) => {
    const by = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map((id) => document.getElementById(id)?.innerText || '').join(' ');
    const labels = el.labels ? [...el.labels].map((l) => l.innerText).join(' ') : '';
    const inputBtn = el.tagName === 'INPUT' && ['button', 'submit', 'reset'].includes(el.type) ? el.value : '';
    const inner = el.tagName === 'INPUT' || el.tagName === 'SELECT' ? '' : el.innerText;
    // Icon-only controls often carry their meaning only in a test hook or id ("shopping-cart-link").
    const hook = clean((el.getAttribute('data-test') || el.getAttribute('data-testid') || el.id || '').replace(/[-_]+/g, ' '), 40);
    const name = clean(
      el.getAttribute('aria-label') || by || labels || inputBtn || inner ||
      el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('alt') ||
      el.querySelector?.('img[alt]')?.getAttribute('alt') ||
      (el.tagName === 'SELECT' ? el.getAttribute('name') || '' : ''),
    );
    // A badge such as "2" alone says nothing about the control; prefix the hook when the text is that short.
    return name.length >= 3 ? name : clean(`${hook} ${name}`) || name;
  };
  const describe = (el, id) => {
    const tag = el.tagName.toLowerCase();
    const attrs = [];
    const role = el.getAttribute('role');
    if (role) attrs.push(`role=${role}`);
    if (tag === 'input') attrs.push(`type=${el.type}`);
    for (const a of ['name', 'placeholder']) {
      const v = el.getAttribute(a);
      if (v) attrs.push(`${a}="${clean(v, 40)}"`);
    }
    if (tag === 'a' && el.hasAttribute('href')) {
      const href = el.getAttribute('href') || '';
      attrs.push(`href="${clean(href.startsWith('javascript:') ? 'javascript' : href, 90)}"`);
    }
    // Never the secret itself, but whether it is filled, so nobody retypes it blindly.
    if (el.type === 'password') attrs.push(el.value ? 'filled' : 'empty');
    if (['input', 'textarea'].includes(tag) && el.type !== 'password' && !['checkbox', 'radio'].includes(el.type)) {
      attrs.push(`value="${clean(el.value, 60)}"`);
    }
    if (el.isContentEditable) attrs.push(`value="${clean(el.innerText, 60)}"`);
    if (['checkbox', 'radio'].includes(el.type)) attrs.push(el.checked ? 'checked' : 'unchecked');
    for (const a of ['aria-checked', 'aria-selected', 'aria-expanded']) {
      const v = el.getAttribute(a);
      if (v !== null) attrs.push(`${a.slice(5)}=${v}`);
    }
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') attrs.push('disabled');
    let label = accessibleName(el);
    if (tag === 'select') {
      const opts = [...el.options].slice(0, 25).map((o) => (o.selected ? `*${clean(o.label, 30)}*` : clean(o.label, 30)));
      label = `${label ? label + ': ' : ''}${opts.join(' | ')}${el.options.length > 25 ? ' | …' : ''}`;
    }
    const r = el.getBoundingClientRect();
    const where = r.bottom < 0 ? ' (above)' : r.top > vh ? ' (below)' : '';
    return `[${id}]<${tag}${attrs.length ? ' ' + attrs.join(' ') : ''}>${label}</${tag}>${where}`;
  };

  // Structured facts for the fast policy (System 1): role, current value/state, options, editability.
  const facts = (el) => {
    const tag = el.tagName;
    const role = el.getAttribute('role') ||
      (tag === 'A' ? 'link' : tag === 'SELECT' ? 'combobox' : tag === 'TEXTAREA' ? 'textbox' :
        tag === 'INPUT' ? (['checkbox', 'radio'].includes(el.type) ? el.type :
          ['button', 'submit', 'reset', 'image'].includes(el.type) ? 'button' : 'textbox') :
          el.isContentEditable ? 'textbox' : 'button');
    const editable = !el.readOnly && !el.disabled && (el.isContentEditable || tag === 'TEXTAREA' ||
      (tag === 'INPUT' && !['checkbox', 'radio', 'button', 'submit', 'reset', 'image', 'file', 'range', 'color']
        .includes(el.type)));
    const f = { role, editable };
    if (el === document.activeElement) f.focused = true;
    if (editable && !el.list && (role === 'combobox' ||
        ['list', 'both'].includes(el.getAttribute('aria-autocomplete')))) {
      f.autocomplete = true;
      if (!autocompleteIds.nodes.has(el)) autocompleteIds.nodes.set(el, autocompleteIds.next++);
      f.autocomplete_id = `${autocompleteIds.document}:${autocompleteIds.nodes.get(el)}`;
    }
    if (el.type !== 'password' && (tag === 'INPUT' || tag === 'TEXTAREA') && editable) f.value = clean(el.value, 80);
    if (el.type === 'password') f.value = el.value ? '(filled, hidden)' : '';
    if (el.isContentEditable) f.value = clean(el.innerText, 80);
    if (['checkbox', 'radio'].includes(el.type)) f.checked = el.checked;
    for (const a of ['aria-checked', 'aria-selected', 'aria-expanded']) {
      const v = el.getAttribute(a);
      if (v !== null) f[a.slice(5)] = v;
    }
    if (tag === 'SELECT') {
      f.options = [...el.options].slice(0, 40).filter((o) => !o.disabled).map((o) => clean(o.label, 40));
      f.value = [...el.selectedOptions].map((o) => clean(o.label, 40)).join(', ');
    }
    if (tag === 'A') f.href = clean(el.href, 120);
    return f;
  };

  const out = [];
  let length = 0, truncated = false, line = '';
  const flush = () => {
    const t = line.trim();
    if (t) { out.push(t); length += t.length + 1; }
    line = '';
    if (length > budget) truncated = true;
  };
  // The budget is a hard cap. One text node or one long inline run can hold a whole document (a raw JSON or text
  // response, thousands of links in one paragraph), so it is checked on every append, not only between blocks.
  const append = (s) => {
    const room = budget - length - line.length;
    if (s.length > room) {
      line += s.slice(0, Math.max(room, 0));
      truncated = true;
      return;
    }
    line += s;
  };
  const elements = [];

  const walk = (node) => {
    if (truncated) return;
    if (node.nodeType === Node.TEXT_NODE) {
      const t = node.textContent.replace(/\s+/g, ' ');
      if (t.trim()) append(t);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    const el = node;
    if (SKIP.has(el.tagName.toUpperCase()) || el.hasAttribute('data-ha-overlay')) return;
    if (el.getAttribute('aria-hidden') === 'true') return;
    const block = BLOCK.test(el.tagName);
    // Styled checkboxes/radios are often hidden behind their label; expose the label as the control.
    const toggle = el.tagName === 'INPUT' && ['checkbox', 'radio'].includes(el.type);
    if (toggle && !visible(el)) {
      const lbl = el.labels?.[0];
      if (lbl && visible(lbl) && hasBox(lbl) && !state.labels?.has(lbl)) {
        (state.labels ||= new Set()).add(lbl);
        const id = state.next++;
        state.nodes.set(id, lbl);
        const r = lbl.getBoundingClientRect();
        const status = el.checked ? 'checked' : 'unchecked';
        elements.push({
          id, tag: 'label', label: clean(lbl.innerText), type: el.type, ...facts(el), editable: false,
          rect: { x: r.x, y: r.y, w: r.width, h: r.height },
        });
        append(` [${id}]<label ${el.type} ${status}>${clean(lbl.innerText)}</label> `);
      }
      return;
    }
    if (el.tagName === 'LABEL' && state.labels?.has(el)) return;
    if (!visible(el) && !(el.tagName === 'SELECT')) {
      // Hidden containers can still hold visible descendants (display: contents); skip only true hides.
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || el.tagName === 'INPUT') return;
      if (cs.display !== 'contents') return;
    }
    if (block) flush();
    const isWrapperWithControls = !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A', 'SUMMARY'].includes(el.tagName)
      && el.querySelector?.(NESTED_CONTROL);
    if (isInteractive(el) && hasBox(el) && !isWrapperWithControls) {
      const id = state.next++;
      state.nodes.set(id, el);
      const r = el.getBoundingClientRect();
      elements.push({
        id, tag: el.tagName.toLowerCase(), label: accessibleName(el) || el.getAttribute('placeholder') || '',
        type: el.type || null, ...facts(el), rect: { x: r.x, y: r.y, w: r.width, h: r.height },
        offscreen: r.bottom < 0 || r.top > vh,
      });
      append(' ' + describe(el, id) + ' ');
      if (block) flush();
      return;
    }
    // Ratings are often drawn with icons and encoded only in a class (e.g. "star-rating Three"); surface them.
    const cls = typeof el.className === 'string' ? el.className : '';
    if (cls.split(/\s+/).some((c) => /rating/i.test(c) || /^stars?$/i.test(c)) && !clean(el.innerText)) append(` [${clean(cls, 60)}] `);
    if (el.tagName === 'IMG') {
      const alt = clean(el.getAttribute('alt'), 60);
      if (alt) append(` [img: ${alt}] `);
    }
    if (el.tagName === 'IFRAME') { flush(); out.push('[iframe content not shown]'); return; }
    for (const child of el.shadowRoot ? el.shadowRoot.childNodes : []) walk(child);
    for (const child of el.childNodes) walk(child);
    if (block) flush();
  };
  walk(document.body || document.documentElement);
  flush();

  // Relate suggestions to their field using observed accessibility ownership, with current observation ids.
  for (const field of elements.filter(e => e.autocomplete)) {
    const node = state.nodes.get(field.id);
    const ids = ((node.getAttribute('aria-controls') || '') + ' ' +
      (node.getAttribute('aria-owns') || '')).split(/\s+/).filter(Boolean);
    const owner = node.closest('[role="combobox"]');
    const roots = ids.length ? ids.map(id => document.getElementById(id)).filter(Boolean) :
      owner && owner !== node ? [owner] : [];
    field.suggestion_ids = elements.filter(e => e.role === 'option' &&
      roots.some(root => root.contains(state.nodes.get(e.id)))).map(e => e.id);
  }

  const height = document.documentElement.scrollHeight;
  return {
    url: location.href,
    title: document.title,
    text: out.join('\n'),
    truncated,
    elements,
    scroll: { y: Math.round(scrollY), viewport: vh, height, width: vw },
  };
}
