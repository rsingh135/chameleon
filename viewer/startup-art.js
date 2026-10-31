/* Original procedural print artwork. The intro controller owns all scheduling. */
(() => {
  'use strict';
  const W = 1100, H = 460, TAU = Math.PI * 2;
  const ink = '#20285b', cyan = '#269fbc', coral = '#ed5848', yellow = '#f4c842', paper = '#f4f0e5';
  const clamp = value => Math.max(0, Math.min(1, value));
  const smooth = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
  const lerp = (a, b, t) => a + (b - a) * t;

  function create(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas is unavailable.');
    let alive = true, scale = 1, offsetX = 0, offsetY = 0;
    let seed = 8423;
    const random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
    const layer = (width, height) => {
      const item = document.createElement('canvas');
      item.width = width; item.height = height;
      return item;
    };
    const texture = layer(192, 192);
    const tc = texture.getContext('2d');
    if (!tc) throw new Error('Print texture is unavailable.');
    // Seeded ink flecks are prepared once, never regenerated during playback.
    for (let i = 0; i < 2700; i++) {
      tc.fillStyle = i % 3 ? '#f4f0e52e' : '#20285b35';
      const size = .35 + random() * 1.25;
      tc.fillRect(random() * 192, random() * 192, size, size);
    }
    const screen = layer(36, 36);
    const sc = screen.getContext('2d');
    if (!sc) throw new Error('Print screen is unavailable.');
    sc.fillStyle = '#20285b';
    for (let y = -6; y <= 36; y += 6) {
      for (let x = -6; x <= 36; x += 6) {
        sc.beginPath(); sc.arc(x + (y % 12 ? 3 : 0), y, 1.1, 0, TAU); sc.fill();
      }
    }
    const grain = ctx.createPattern(texture, 'repeat');
    const dots = ctx.createPattern(screen, 'repeat');
    const background = layer(W, H);
    const bg = background.getContext('2d');
    if (!bg) throw new Error('Print background is unavailable.');

    function circle(c, x, y, radius, fill) {
      c.beginPath(); c.arc(x, y, radius, 0, TAU); c.fillStyle = fill; c.fill();
    }
    function leaf(c, x, y, length, width, angle, fill) {
      c.save(); c.translate(x, y); c.rotate(angle);
      c.beginPath(); c.moveTo(0, 0);
      c.bezierCurveTo(length * .26, -width, length * .86, -width * .76, length, 0);
      c.bezierCurveTo(length * .66, width * .49, length * .27, width * .63, 0, 0);
      c.fillStyle = fill; c.fill();
      c.save(); c.clip(); c.globalAlpha = .22; c.fillStyle = dots; c.fillRect(-4, -width, length + 8, width * 2); c.restore();
      c.strokeStyle = paper; c.lineWidth = 1.4; c.globalAlpha = .65;
      c.beginPath(); c.moveTo(6, 0); c.quadraticCurveTo(length * .55, -width * .13, length - 15, 0); c.stroke();
      c.restore();
    }
    // A small print composition: one sun, two botanical accents and a branch.
    circle(bg, 601, 200, 157, yellow);
    bg.save(); bg.beginPath(); bg.arc(601, 200, 157, 0, TAU); bg.clip();
    bg.globalAlpha = .2; bg.fillStyle = bg.createPattern(screen, 'repeat'); bg.fillRect(440, 40, 325, 325);
    bg.globalAlpha = .6; bg.strokeStyle = paper; bg.lineWidth = 5;
    bg.beginPath(); bg.arc(606, 197, 150, Math.PI * .96, Math.PI * 1.62); bg.stroke(); bg.restore();
    circle(bg, 305, 122, 23, coral);
    bg.strokeStyle = ink; bg.lineWidth = 2;
    bg.beginPath(); bg.moveTo(129, 393); bg.bezierCurveTo(230, 365, 455, 371, 986, 366); bg.stroke();
    bg.lineWidth = 1.3;
    bg.beginPath(); bg.moveTo(152, 386); bg.quadraticCurveTo(166, 329, 124, 248); bg.stroke();
    leaf(bg, 153, 351, 81, 27, -2.12, cyan);
    leaf(bg, 151, 337, 85, 26, -1.06, ink);
    leaf(bg, 144, 308, 65, 21, -2.22, cyan);
    leaf(bg, 131, 274, 48, 17, -1.36, coral);
    bg.beginPath(); bg.moveTo(948, 367); bg.quadraticCurveTo(963, 321, 1001, 294); bg.stroke();
    leaf(bg, 960, 341, 72, 24, -1.75, cyan);
    leaf(bg, 974, 321, 68, 23, -.46, coral);
    leaf(bg, 986, 307, 52, 19, -1.23, ink);
    bg.save(); bg.fillStyle = bg.createPattern(texture, 'repeat'); bg.globalAlpha = .42; bg.fillRect(70, 35, 970, 370); bg.restore();

    function body() {
      ctx.beginPath(); ctx.moveTo(-4, -100);
      ctx.bezierCurveTo(6, -135, 33, -177, 78, -188);
      ctx.bezierCurveTo(132, -205, 191, -189, 224, -158);
      ctx.bezierCurveTo(246, -131, 250, -85, 220, -67);
      ctx.bezierCurveTo(184, -45, 92, -48, 48, -63);
      ctx.bezierCurveTo(15, -74, 4, -86, -4, -100); ctx.closePath();
    }
    function head() {
      ctx.beginPath(); ctx.moveTo(208, -136); ctx.lineTo(226, -171);
      ctx.quadraticCurveTo(239, -188, 251, -218);
      ctx.quadraticCurveTo(266, -210, 269, -181);
      ctx.bezierCurveTo(275, -173, 287, -173, 297, -165);
      ctx.quadraticCurveTo(312, -148, 315, -127);
      ctx.quadraticCurveTo(314, -113, 298, -110);
      ctx.lineTo(264, -108); ctx.quadraticCurveTo(234, -90, 216, -112); ctx.closePath();
    }
    function tail(t, changed) {
      const sway = Math.sin(Math.min(t, 1.9) * 6) * 2.6 * (1 - smooth((t - 1.65) / .3));
      ctx.save(); ctx.translate(0, sway);
      ctx.beginPath(); ctx.moveTo(18, -107);
      ctx.bezierCurveTo(-38, -137, -141, -145, -173, -100);
      for (let i = 0; i <= 104; i++) {
        const progress = i / 104, angle = -2.9 + progress * TAU * 1.55;
        const radius = lerp(67, 11, progress);
        ctx.lineTo(-108 + Math.cos(angle) * radius, -84 + Math.sin(angle) * radius);
      }
      ctx.lineWidth = 21; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.strokeStyle = changed > .65 ? coral : cyan; ctx.stroke();
      ctx.lineWidth = 6; ctx.strokeStyle = ink; ctx.globalAlpha = .28; ctx.stroke();
      ctx.lineWidth = 21; ctx.strokeStyle = grain; ctx.globalAlpha = 1; ctx.stroke(); ctx.restore();
    }
    function leg(x, phase, near, t, changed) {
      const walking = 1 - smooth((t - 1.55) / .35);
      const step = Math.sin(phase) * walking;
      const lift = Math.max(0, Math.cos(phase)) * 16 * walking;
      const footX = x + 9 + step * 23, footY = -lift;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = near ? 15 : 11;
      ctx.strokeStyle = near ? (changed > .4 ? coral : cyan) : ink;
      ctx.beginPath(); ctx.moveTo(x, -72);
      ctx.quadraticCurveTo(x - 24, -46, x - 13 + step * 8, -30);
      ctx.lineTo(footX, footY - 4); ctx.stroke();
      ctx.lineWidth = near ? 5 : 4;
      for (let toe = 0; toe < 3; toe++) {
        const tip = footX + 9 + toe * 6;
        ctx.beginPath(); ctx.moveTo(footX - 4, footY - 5);
        ctx.quadraticCurveTo(tip + 5, footY - 6 + toe, tip, footY + 4);
        ctx.quadraticCurveTo(tip - 4, footY + 9, tip - 8, footY + 4); ctx.stroke();
      }
      if (near) { circle(ctx, x - 12 + step * 8, -31, 3.8, yellow); }
    }
    function printBody(t, changed) {
      body(); ctx.fillStyle = cyan; ctx.fill();
      ctx.save(); body(); ctx.clip();
      // The changing ink moves from the mouth toward the tail.
      if (changed > 0) {
        ctx.save(); ctx.beginPath(); ctx.rect(330 - changed * 570, -245, 600, 250); ctx.clip();
        ctx.fillStyle = coral; ctx.fillRect(-40, -230, 380, 250);
        ctx.fillStyle = yellow; ctx.beginPath(); ctx.moveTo(41, -234); ctx.bezierCurveTo(70, -144, 119, -130, 109, -20); ctx.lineTo(164, -20); ctx.bezierCurveTo(170, -125, 119, -179, 126, -234); ctx.closePath(); ctx.fill();
        ctx.fillStyle = cyan; ctx.beginPath(); ctx.moveTo(170, -230); ctx.bezierCurveTo(149, -151, 233, -141, 203, -25); ctx.lineTo(260, -25); ctx.lineTo(260, -230); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      ctx.fillStyle = ink; ctx.globalAlpha = .9;
      for (let i = 0; i < 5; i++) {
        ctx.beginPath(); ctx.moveTo(28 + i * 39, -204);
        ctx.bezierCurveTo(19 + i * 40, -165, 57 + i * 36, -153, 41 + i * 39, -109);
        ctx.bezierCurveTo(71 + i * 36, -137, 42 + i * 40, -167, 51 + i * 39, -203); ctx.fill();
      }
      ctx.globalAlpha = .2; ctx.fillStyle = dots; ctx.fillRect(-20, -220, 300, 190);
      ctx.globalAlpha = 1; ctx.fillStyle = paper;
      for (let i = 0; i < 11; i++) { circle(ctx, 22 + i * 19, -92 + Math.sin(i * .46) * 8, i % 3 ? 2.2 : 3.2, paper); }
      ctx.globalAlpha = .85; ctx.fillStyle = grain; ctx.fillRect(-20, -220, 310, 190); ctx.restore();
      // A faint second plate gives the silhouette an intentional print offset.
      ctx.save(); ctx.translate(2, -1); body(); ctx.strokeStyle = ink; ctx.globalAlpha = .7; ctx.lineWidth = 1.8; ctx.stroke(); ctx.restore();
      ctx.fillStyle = changed > .3 ? coral : cyan;
      for (let i = 0; i < 12; i++) {
        const x = 27 + i * 15.4, y = -181 - Math.sin((i / 12) * Math.PI) * 14;
        ctx.beginPath(); ctx.moveTo(x - 4, y + 3); ctx.lineTo(x + 2, y - 7); ctx.lineTo(x + 7, y + 2); ctx.fill();
      }
    }
    function printHead(t, changed, gazeX, gazeY) {
      const look = smooth((t - 1.9) / .45);
      const proud = smooth((t - 3.2) / .6);
      ctx.save(); ctx.translate(224, -126); ctx.rotate(-look * .045 - proud * .035); ctx.translate(-224, 126);
      head(); ctx.fillStyle = changed > .12 ? coral : cyan; ctx.fill();
      ctx.save(); head(); ctx.clip(); ctx.fillStyle = yellow;
      ctx.beginPath(); ctx.moveTo(236, -223); ctx.quadraticCurveTo(249, -159, 221, -115); ctx.lineTo(240, -111); ctx.quadraticCurveTo(278, -160, 254, -225); ctx.fill();
      ctx.globalAlpha = .2; ctx.fillStyle = dots; ctx.fillRect(210, -230, 110, 130);
      ctx.globalAlpha = 1; ctx.fillStyle = grain; ctx.fillRect(210, -230, 110, 130); ctx.restore();
      ctx.strokeStyle = ink; ctx.lineWidth = 1.8; head(); ctx.stroke();
      circle(ctx, 276, -154, 22, yellow); circle(ctx, 276, -154, 17, paper);
      ctx.strokeStyle = ink; ctx.lineWidth = 1.7; ctx.beginPath(); ctx.arc(276, -154, 22, 0, TAU); ctx.stroke();
      const blink = t > 4.1 && t < 4.26 ? Math.sin(((t - 4.1) / .16) * Math.PI) : 0;
      ctx.save(); ctx.translate(277 + gazeX * 5, -155 + gazeY * 5); ctx.scale(1, Math.max(.08, 1 - blink));
      circle(ctx, 0, 0, 8, ink); circle(ctx, 2, -3, 2.2, paper); ctx.restore();
      circle(ctx, 305, -137, 1.8, ink);
      ctx.beginPath(); ctx.moveTo(281, -119); ctx.quadraticCurveTo(297, -115, 313, -126); ctx.stroke();
      ctx.restore();
    }
    function swatch(x, y, size, angle) {
      ctx.save(); ctx.translate(x, y); ctx.rotate(angle); ctx.scale(size, size);
      ctx.fillStyle = coral; ctx.fillRect(-17, -21, 34, 42);
      ctx.fillStyle = yellow; ctx.fillRect(-17, 8, 34, 13);
      ctx.fillStyle = ink; ctx.globalAlpha = .25; ctx.fillRect(-20, -18, 2, 40); ctx.globalAlpha = .23;
      ctx.fillStyle = dots; ctx.fillRect(-17, -21, 34, 42); ctx.globalAlpha = 1;
      ctx.fillStyle = grain; ctx.fillRect(-17, -21, 34, 42); ctx.restore();
    }
    function burst(t) {
      const p = clamp((t - 2.84) / .6);
      if (!p || p === 1) return;
      ctx.save(); ctx.globalAlpha = Math.sin(p * Math.PI) * .9;
      for (let i = 0; i < 9; i++) {
        const angle = i * TAU / 9 + .2, radius = 23 + p * 61;
        ctx.save(); ctx.translate(809 + Math.cos(angle) * radius, 180 + Math.sin(angle) * radius); ctx.rotate(angle);
        ctx.fillStyle = [coral, cyan, yellow][i % 3]; ctx.fillRect(-3, -2, 8 + 6 * (1 - p), 4); ctx.restore();
      }
      ctx.restore();
    }
    function resize() {
      if (!alive) return;
      const box = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(box.width * dpr));
      const height = Math.max(1, Math.round(box.height * dpr));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      scale = Math.min(width / W, height / H);
      offsetX = (width - W * scale) / 2; offsetY = (height - H * scale) / 2;
    }
    function draw(seconds) {
      if (!alive) return;
      const t = Math.max(0, Math.min(5.2, Number(seconds) || 0));
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
      ctx.drawImage(background, 0, 0);
      const walk = clamp(t / 1.9);
      const travel = walk < .82 ? walk / .91 : lerp(.901, 1, smooth((walk - .82) / .18));
      const x = lerp(-290, 435, Math.min(1, travel));
      const gait = Math.min(t, 1.9) * 10.8;
      const bob = Math.sin(gait * 2) * 2.3 * (1 - smooth((t - 1.55) / .35));
      const changed = smooth((t - 3) / 1.04);
      const swatchX = 809 + Math.sin(t * 1.8) * 7;
      const swatchY = 180 + Math.cos(t * 2.2) * 8;
      if (t < 2.85) swatch(swatchX, swatchY, 1, -.19 + Math.sin(t * 1.5) * .13);
      // Tongue shoots out, then the captured print comes back toward the mouth.
      if (t > 2.6 && t < 3.06) {
        const snap = t < 2.84 ? smooth((t - 2.6) / .24) : 1 - smooth((t - 2.84) / .22);
        const mouthX = x + 313, mouthY = 367 + bob - 128;
        const tipX = lerp(mouthX, swatchX, snap), tipY = lerp(mouthY, swatchY, snap);
        ctx.strokeStyle = coral; ctx.lineWidth = 5; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(mouthX, mouthY); ctx.quadraticCurveTo(lerp(mouthX, tipX, .6), mouthY - snap * 15, tipX, tipY); ctx.stroke();
        circle(ctx, tipX, tipY, 5.5, coral);
        if (t >= 2.85) swatch(tipX, tipY, Math.max(.15, snap), -.3 + (1 - snap) * .7);
      }
      burst(t);
      ctx.save(); ctx.translate(x, 367 + bob);
      leg(52, gait + Math.PI, false, t, changed); leg(190, gait, false, t, changed);
      tail(t, changed); printBody(t, changed);
      leg(35, gait, true, t, changed); leg(183, gait + Math.PI, true, t, changed);
      printHead(t, changed, t < 1.9 ? .65 : .85, t < 1.9 ? 0 : -.7);
      ctx.restore();
      // Registration marks quietly frame the illustration, not the working UI.
      ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.globalAlpha = .45;
      for (const [cx, cy] of [[93, 74], [1010, 397]]) {
        ctx.beginPath(); ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy); ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    function dispose() {
      alive = false;
      background.width = background.height = texture.width = texture.height = screen.width = screen.height = 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    return {draw, resize, dispose};
  }
  window.ChameleonIntroArt = {create};
})();
