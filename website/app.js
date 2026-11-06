(() => {
  const canvas = document.getElementById('neural-field');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const points = Array.from({ length: 88 }, (_, index) => {
    const a = (index * 2.399963229728653) % (Math.PI * 2);
    const r = Math.sqrt((index + 0.5) / 88);
    return { a, r, phase: index * 2.19, speed: 0.35 + (index % 7) * 0.10 };
  });
  let t = 0;
  let last = 0;
  function draw(timestamp) {
    const w = canvas.width, h = canvas.height;
    const cx = w * .51, cy = h * .49;
    const rX = 288, rY = 283;
    ctx.clearRect(0, 0, w, h);
    const ring = ctx.createRadialGradient(cx, cy, 55, cx, cy, 340);
    ring.addColorStop(0, 'rgba(105,255,180,0)');
    ring.addColorStop(.62, 'rgba(78,196,175,.055)');
    ring.addColorStop(1, 'rgba(131,159,255,0)');
    ctx.fillStyle = ring;
    ctx.beginPath(); ctx.arc(cx, cy, 337, 0, 2*Math.PI); ctx.fill();
    for (let i = 0; i < 11; i++) {
      ctx.beginPath();
      const yy = cy + (i - 5) * 20;
      for (let x = 30; x < w - 15; x += 3) {
        const influence = Math.exp(-(((x-cx)/280)**2));
        const amp = 6 + 13 * influence;
        const y = yy + (Math.sin(x * .029 + t * .48 + i*.7) + .35*Math.sin(x*.085 - t*.62+i)) * amp * .45;
        if (x === 30) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.strokeStyle = i % 3 ? 'rgba(95,183,173,.08)' : 'rgba(159,242,178,.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const twist = Math.sin(t * .2 + p.phase) * .04;
      const x = cx + Math.cos(p.a + twist) * rX * p.r;
      const y = cy + Math.sin(p.a + twist) * rY * p.r * .90;
      p.x = x; p.y = y;
      const alpha = .2 + .6 * (.5 + .5 * Math.sin(t * p.speed + p.phase));
      ctx.beginPath(); ctx.arc(x,y,i % 13===0 ? 3.0 : 1.55,0,Math.PI*2);
      ctx.fillStyle = `rgba(156,255,189,${alpha*.6})`; ctx.fill();
      if (i % 6 === 0) {
        ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(cx,cy);
        ctx.strokeStyle = `rgba(110,230,182,${alpha*.048})`; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    for (let i = 0; i < points.length; i += 3) {
      const p = points[i];
      const q = points[(i + 21) % points.length];
      if (Math.hypot(p.x-q.x,p.y-q.y) > 245) continue;
      ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
      ctx.strokeStyle = 'rgba(150,216,191,.065)';ctx.lineWidth=1;ctx.stroke();
    }
    if (!reduced) {
      t += Math.min(33, timestamp - last) / 1000 || .015;
      last = timestamp;
      requestAnimationFrame(draw);
    }
  }
  draw(0);
})();
