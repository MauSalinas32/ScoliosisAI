/*
 * Accessibility audit of the web portal (automated WCAG 2.1 A/AA rules with axe-core + a keyboard-only login).
 *
 *   uvicorn main:app --port 8000 > server.log 2>&1 &
 *   npm install playwright axe-core        # one time, in a scratch folder
 *   BASE=http://localhost:8000 LOG=server.log OUT=evidence/accessibility.json node scripts/accessibility/audit.js
 *
 * Automated rules find only part of the real barriers (roughly a third); a manual review with a screen reader
 * is still required. The JSON lists every violation with its impact so that it can be tracked.
 */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const BASE = process.env.BASE || 'http://localhost:8000';
const LOG = process.env.LOG || 'server.log';
const OUT = process.env.OUT || 'accessibility.json';
const AXE = require.resolve('axe-core/axe.min.js');
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

function lastCode(user) {
  const m = [...fs.readFileSync(LOG, 'utf8').matchAll(new RegExp(`CODE for '${user}': (\\d{6})`, 'g'))];
  return m[m.length - 1][1];
}

async function scan(page, label, results) {
  await page.addScriptTag({ path: AXE });
  const r = await page.evaluate(async (tags) => await axe.run(document, { runOnly: { type: 'tag', values: tags } }), TAGS);
  results.push({
    page: label,
    violations: r.violations.map(v => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.length, targets: v.nodes.slice(0, 6).map(n => n.target.join(' ')) })),
    passes: r.passes.length,
  });
}

async function login(browser, user, withKeyboard) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await ctx.newPage();
  await page.goto(BASE);
  await page.waitForSelector('#login-user');
  return { ctx, page };
}

(async () => {
  const browser = await chromium.launch();
  const results = [];
  const keyboard = {};

  // 1. login page
  let { page } = await login(browser, 'technician');
  await scan(page, 'login', results);

  // 2. keyboard-only login (no mouse): Tab to the fields, type, Enter
  await page.focus('#login-user');
  await page.keyboard.type('radiologist');
  await page.keyboard.press('Tab');
  await page.keyboard.type('password123');
  await page.keyboard.press('Enter');
  await page.waitForSelector('#two-factor-form', { state: 'visible' });
  await page.waitForTimeout(800);
  await scan(page, '2fa', results);
  await page.focus('#tfa-code');
  await page.keyboard.type(lastCode('radiologist'));
  await page.keyboard.press('Enter');
  keyboard.login_with_keyboard_only = await page.waitForSelector('#nav-img', { state: 'visible', timeout: 8000 }).then(() => true).catch(() => false);

  // 3. can the navigation be reached and activated with the keyboard?
  const navIds = ['nav-img', 'nav-ai', 'nav-results'];
  keyboard.navigation = {};
  for (const id of navIds) {
    const focusable = await page.evaluate((i) => { const el = document.getElementById(i); return !!el && el.tabIndex >= 0; }, id);
    await page.focus(`#${id}`).catch(() => {});
    await page.keyboard.press('Enter');
    await page.waitForTimeout(300);
    const active = await page.evaluate((i) => document.getElementById(i).classList.contains('active'), id);
    keyboard.navigation[id] = { focusable_by_tab: focusable, activated_with_enter: active };
  }
  await scan(page, 'results (radiologist)', results);
  await page.click('#nav-ai');
  await scan(page, 'ai tab (radiologist)', results);

  // 4. other roles
  for (const [user, nav, label] of [['technician', null, 'upload (technician)'], ['admin', 'nav-audit', 'audit (admin)']]) {
    const s = await login(browser, user);
    await s.page.fill('#login-user', user);
    await s.page.fill('#login-pass', 'password123');
    await s.page.click('#btn-login');
    await s.page.waitForSelector('#two-factor-form', { state: 'visible' });
    await s.page.waitForTimeout(800);
    await s.page.fill('#tfa-code', lastCode(user));
    await s.page.click('#btn-verify-2fa');
    await s.page.waitForSelector('#nav-img', { state: 'visible' });
    if (nav) await s.page.click(`#${nav}`);
    await scan(s.page, label, results);
  }

  const summary = {};
  for (const r of results) for (const v of r.violations) {
    summary[v.id] = summary[v.id] || { impact: v.impact, help: v.help, pages: [], nodes: 0 };
    summary[v.id].pages.push(r.page); summary[v.id].nodes += v.nodes;
  }
  fs.writeFileSync(OUT, JSON.stringify({ rules: TAGS, keyboard, summary, per_page: results }, null, 2));
  console.log(JSON.stringify({ keyboard, violations: Object.fromEntries(Object.entries(summary).map(([k, v]) => [k, `${v.impact}, ${v.nodes} nodes`])) }, null, 2));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
