/* SmartWater AI — shared UI helpers & premium micro-interactions */
(function () {
  "use strict";

  var prefersReduced = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;

  // ---------- number formatting ----------
  window.fmt = function (v) {
    return Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
  };
  window.fmt1 = function (v) {
    return Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });
  };

  // ---------- demand level badges ----------
  window.levelBadge = function (level, extra) {
    const key = String(level || "").toUpperCase().replace(/\s+/g, "-");
    const cls = { LOW: "lvl-low", NORMAL: "lvl-normal", HIGH: "lvl-high", "VERY-HIGH": "lvl-very-high" }[key] || "lvl-normal";
    return `<span class="badge ${cls} ${extra || ""}">${level || "—"}</span>`;
  };

  // ---------- mobile nav ----------
  window.initNav = function () {
    const toggle = document.getElementById("nav-toggle");
    const links = document.getElementById("nav-links");
    if (!toggle || !links) return;
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.innerHTML = open
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
    });
    links.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
        toggle.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
      });
    });
    // nav solidifies on scroll
    const bar = document.querySelector(".topbar");
    if (bar) {
      const onScroll = function () { bar.classList.toggle("scrolled", window.scrollY > 24); };
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }
  };

  // ---------- animated number count-up ----------
  window.countUp = function (el, target, opts) {
    if (!el) return;
    opts = opts || {};
    const duration = opts.duration || 1100;
    const decimals = opts.decimals != null ? opts.decimals : 0;
    const start = performance.now();
    target = Number(target) || 0;

    if (prefersReduced) {
      el.textContent = Number(target).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
      return;
    }
    const fmtNum = function (v) {
      return v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    };
    const tick = function (now) {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmtNum(target * eased);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  // ---------- scroll reveal ----------
  window.initReveals = function () {
    const els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (!("IntersectionObserver" in window) || prefersReduced) {
      els.forEach(function (el) { el.classList.add("in-view"); });
      return;
    }
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("in-view");
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    els.forEach(function (el) { io.observe(el); });
  };

  // ---------- scroll progress bar ----------
  window.initProgress = function () {
    const bar = document.querySelector(".progress-bar span");
    if (!bar) return;
    const update = function () {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const p = max > 0 ? (window.scrollY / max) * 100 : 0;
      bar.style.width = p.toFixed(2) + "%";
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
  };

  // ---------- slide rail (dashboard section dots) ----------
  window.initSlideRail = function () {
    const rail = document.getElementById("slide-rail");
    if (!rail) return;
    const slides = Array.prototype.slice.call(document.querySelectorAll(".slide[id]"));
    if (!slides.length) return;

    slides.forEach(function (s) {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-label", "Go to " + s.dataset.label);
      b.innerHTML = '<span class="lbl">' + (s.dataset.label || "") + "</span><span class='dot'></span>";
      b.addEventListener("click", function () {
        s.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth", block: "start" });
      });
      rail.appendChild(b);
    });

    const btns = Array.prototype.slice.call(rail.querySelectorAll("button"));
    const map = {};
    slides.forEach(function (s, i) { map[s.id] = btns[i]; });

    let activeId = slides[0].id;
    const setActive = function (id) {
      if (id === activeId) return;
      activeId = id;
      btns.forEach(function (b) { b.classList.remove("on"); });
      const b = map[id];
      if (b) b.classList.add("on");
    };

    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) setActive(e.target.id);
        });
      }, { threshold: 0.55 });
      slides.forEach(function (s) { io.observe(s); });
    } else {
      const onScroll = function () {
        let cur = slides[0].id;
        slides.forEach(function (s) { if (window.scrollY >= s.offsetTop - window.innerHeight * 0.5) cur = s.id; });
        setActive(cur);
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }
  };

  // ---------- dashboard section nav (click nav link -> scroll to slide) ----------
  window.initSectionNav = function () {
    const links = document.querySelectorAll("#nav-links a[data-scroll]");
    if (!links.length) return;
    links.forEach(function (a) {
      a.addEventListener("click", function (e) {
        const target = document.getElementById(a.dataset.scroll);
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth", block: "start" });
        // update active state
        document.querySelectorAll("#nav-links a").forEach(function (x) { x.classList.remove("active"); });
        a.classList.add("active");
      });
    });

    // jump to ?s=section on load (cross-page deep links)
    const s = new URLSearchParams(window.location.search).get("s");
    if (s) {
      const target = document.getElementById("sec-" + s) || document.getElementById(s);
      if (target) {
        window.setTimeout(function () {
          target.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth", block: "start" });
        }, 60);
      }
    }
  };

  // ---------- button press ripple ----------
  window.initRipples = function () {
    if (prefersReduced) return;
    document.addEventListener("pointerdown", function (e) {
      const btn = e.target.closest(".btn");
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      const d = Math.max(rect.width, rect.height);
      const ink = document.createElement("span");
      ink.className = "ripple-ink";
      ink.style.width = ink.style.height = d + "px";
      ink.style.left = e.clientX - rect.left - d / 2 + "px";
      ink.style.top = e.clientY - rect.top - d / 2 + "px";
      btn.appendChild(ink);
      window.setTimeout(function () { ink.remove(); }, 650);
    });
  };

  // ---------- chart palette (luxury water-intelligence) ----------
  window.CHART = {
    accent: "#0055a0",
    accentSoft: "rgba(0, 85, 160, 0.13)",
    blueberry: "#0055a0",
    bluebird: "#438bc4",
    sky: "#8cc1e9",
    navy: "#12284b",
    navySoft: "rgba(18, 40, 75, 0.4)",
    barSoft: "rgba(18, 40, 75, 0.55)",
    line: "rgba(140, 193, 233, 0.5)",
    muted: "#52708a",
    ink: "#12284b",
    grid: "rgba(18, 40, 75, 0.07)",
    levels: {
      LOW: "#0f7a58",
      NORMAL: "#0055a0",
      HIGH: "#a16207",
      "VERY HIGH": "#bc2f44",
    },
  };

  window.setChartDefaults = function () {
    if (typeof Chart === "undefined") return;
    Chart.defaults.font.family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = window.CHART.muted;
    Chart.defaults.borderColor = window.CHART.line;
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(18, 40, 75, 0.95)";
    Chart.defaults.plugins.tooltip.titleColor = "#fff";
    Chart.defaults.plugins.tooltip.bodyColor = "rgba(255,255,255,0.88)";
    Chart.defaults.plugins.tooltip.padding = 11;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.tooltip.titleFont = { weight: 600 };
    // merge (never replace) so Chart.js keeps its color/number animation type
    // descriptors — replacing the object wipes them and crashes on color changes
    Chart.defaults.animation = Object.assign({}, Chart.defaults.animation || {}, { duration: 850, easing: "easeOutQuart" });
  };

  // common axis config for a clean look
  window.axisOpts = function () {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: 10, maxRotation: 0 },
        },
        y: {
          beginAtZero: false,
          grid: { color: window.CHART.grid },
          ticks: { maxTicksLimit: 6 },
        },
      },
    };
  };

  document.addEventListener("DOMContentLoaded", function () {
    window.setChartDefaults();
    window.initNav();
    window.initReveals();
    window.initProgress();
    window.initSlideRail();
    window.initSectionNav();
    window.initRipples();
  });
})();