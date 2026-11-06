(() => {
  const get = (id) => document.getElementById(id);
  const raw = get('raw-range');
  const offset = get('offset-range');
  const numeric = get('offset-number');
  if (!raw || !offset || !numeric) return;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const signed = (x) => (x >= 0 ? '+' : '') + x;
  const draw = () => {
    const r = clamp(Number(raw.value) || 0, -128, 127);
    const off = clamp(Math.trunc(Number(numeric.value) || 0), -32768, 32767);
    const sum = r + off;
    const out = clamp(sum, -128, 127);
    const saturation = sum !== out;
    get('raw-value').textContent = signed(r);
    get('offset-value').textContent = signed(off);
    get('raw-preview').textContent = signed(r);
    get('offset-preview').textContent = signed(off);
    get('cal-preview').textContent = signed(out);
    get('cal-track-fill').style.width = `${(out + 128) / 255 * 100}%`;
    get('cal-status').textContent = saturation
      ? `Saturation · integer ${signed(sum)} clipped to ${signed(out)}`
      : 'No saturation · exact integer addition';
    get('cal-status').dataset.saturated = saturation ? 'true' : 'false';
  };
  raw.addEventListener('input', draw);
  offset.addEventListener('input', () => {numeric.value = offset.value; draw();});
  numeric.addEventListener('input', () => {
    const val = clamp(Math.trunc(Number(numeric.value) || 0), -32768, 32767);
    offset.value = String(clamp(val, -512, 512));
    draw();
  });
  get('lab-reset').addEventListener('click', () => {raw.value='48'; offset.value='32'; numeric.value='32';draw();});
  draw();
})();
