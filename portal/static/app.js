/* The workbench's browser side. Four small jobs, no framework:
   tabs, a file box, polling a running job, and redrawing one picture when a control changes. */
(function () {
  "use strict";

  /* ------------------------------------------------------------ tabs */
  function initTabs(root) {
    root.querySelectorAll(".tabs").forEach(function (bar) {
      var buttons = Array.prototype.slice.call(
        bar.querySelectorAll("button[data-panel]"),
      );
      if (!buttons.length) return;
      var group = bar.dataset.group || "";
      function show(name, remember) {
        buttons.forEach(function (b) {
          var on = b.dataset.panel === name;
          b.setAttribute("aria-selected", on ? "true" : "false");
          var panel = document.getElementById(b.dataset.panel);
          if (panel) panel.hidden = !on;
          if (on && panel) loadLazy(panel);
        });
        if (remember && group) {
          try {
            sessionStorage.setItem("tab:" + group, name);
          } catch (e) {
            /* private window */
          }
        }
      }
      buttons.forEach(function (b) {
        b.addEventListener("click", function () {
          show(b.dataset.panel, true);
        });
      });
      var saved = null;
      try {
        saved = group ? sessionStorage.getItem("tab:" + group) : null;
      } catch (e) {
        saved = null;
      }
      var start =
        buttons.filter(function (b) {
          return b.dataset.panel === saved;
        })[0] || buttons[0];
      show(start.dataset.panel, false);
    });
  }

  /* A panel whose contents are fetched the first time it is opened, so a page with twelve maps
     does not make the server draw twelve maps before showing anything. */
  function loadLazy(panel) {
    var url = panel.dataset.lazy;
    if (!url || panel.dataset.loaded === "1") return;
    panel.dataset.loaded = "1";
    panel.innerHTML =
      '<p class="muted small"><span class="spin"></span> Drawing…</p>';
    fetch(url, { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("The server answered " + r.status + ".");
        return r.text();
      })
      .then(function (html) {
        panel.innerHTML = html;
        initTabs(panel);
        initLazy(panel);
        initFigBars(panel);
        initPanelForms(panel);
        initFields(panel);
        drawPlotly(panel);
      })
      .catch(function (e) {
        panel.dataset.loaded = "";
        panel.innerHTML =
          '<p class="note err">This could not be drawn: ' +
          escapeHtml(e.message) +
          ' <button class="btn sec sm" onclick="location.reload()">Try again</button></p>';
      });
  }

  /* A lazily-loaded block that is not inside a tab: fetch it as soon as the page is ready. */
  function initLazy(root) {
    root.querySelectorAll("[data-lazy]").forEach(function (el) {
      if (!el.classList.contains("panel")) loadLazy(el);
    });
  }

  /* ------------------------------------------------------------ plotly */
  function drawPlotly(root) {
    if (typeof Plotly === "undefined") return;
    root.querySelectorAll("script.plotly-spec").forEach(function (tag) {
      if (tag.dataset.drawn === "1") return;
      var el = document.getElementById(tag.dataset.target);
      if (!el) return;
      tag.dataset.drawn = "1";
      var spec = JSON.parse(tag.textContent);
      Plotly.newPlot(el, spec.data, spec.layout, {
        responsive: true,
        displaylogo: false,
        toImageButtonOptions: { scale: 2 },
      });
    });
  }

  /* ------------------------------------------------------------ the file box */
  function initFileBoxes(root) {
    root.querySelectorAll(".filebox").forEach(function (box) {
      var input = box.querySelector("input[type=file]");
      if (!input) return;
      ["dragenter", "dragover"].forEach(function (ev) {
        box.addEventListener(ev, function (e) {
          e.preventDefault();
          box.classList.add("over");
        });
      });
      ["dragleave", "drop"].forEach(function (ev) {
        box.addEventListener(ev, function (e) {
          e.preventDefault();
          box.classList.remove("over");
        });
      });
      box.addEventListener("drop", function (e) {
        if (e.dataTransfer && e.dataTransfer.files.length) {
          input.files = e.dataTransfer.files;
          input.dispatchEvent(new Event("change", { bubbles: true }));
        }
      });
      input.addEventListener("change", function () {
        var form = input.closest("form[data-reconfigure]");
        if (form && input.files.length) form.submit();
      });
    });
  }

  /* ------------------------------------------------------------ a running job */
  function initJob(root) {
    var box = root.querySelector("[data-job]");
    if (!box) return;
    var id = box.dataset.job;
    var fill = box.querySelector(".bar > i");
    var bar = box.querySelector(".bar");
    var say = box.querySelector(".say .tx");
    var el = box.querySelector(".say .el");
    var t0 = Date.now();
    var misses = 0;

    function tick() {
      if (el) el.textContent = Math.round((Date.now() - t0) / 1000) + "s";
    }
    var clock = setInterval(tick, 1000);

    function poll() {
      fetch(box.dataset.poll, { credentials: "same-origin" })
        .then(function (r) {
          if (!r.ok) throw new Error("status " + r.status);
          return r.json();
        })
        .then(function (j) {
          misses = 0;
          if (fill) fill.style.width = Math.max(2, j.progress * 100) + "%";
          if (say) say.textContent = j.message || "Working…";
          if (j.state === "done") {
            clearInterval(clock);
            if (say) say.textContent = "Finished. Loading the results…";
            location.href = box.dataset.done;
          } else if (j.state === "failed" || j.state === "cancelled") {
            clearInterval(clock);
            if (bar) bar.classList.add("err");
            if (fill) fill.style.width = "100%";
            if (say) {
              say.innerHTML =
                (j.state === "cancelled"
                  ? "Cancelled. "
                  : "It did not finish: ") +
                escapeHtml(j.error || "") +
                ' <a href="' +
                box.dataset.retry +
                '">Change the settings and try again</a>.';
            }
          } else {
            setTimeout(poll, 1200);
          }
        })
        .catch(function () {
          misses += 1;
          if (misses > 20) {
            clearInterval(clock);
            if (say)
              say.innerHTML =
                "Lost contact with the server. " +
                '<a href="javascript:location.reload()">Reload</a> to see where it got to.';
            return;
          }
          setTimeout(poll, 2500);
        });
    }
    tick();
    poll();
    void id;
  }

  /* ------------------------------------------------------------ redraw one picture */
  /* A control that only changes how a map is drawn (Normal / Filled, a palette) asks the server for
     that one picture instead of running everything again. */
  function initFigBars(root) {
    root.querySelectorAll("[data-redraw]").forEach(function (bar) {
      if (bar.dataset.wired === "1") return;
      bar.dataset.wired = "1";
      bar.querySelectorAll("button[data-value]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          bar.querySelectorAll("button[data-value]").forEach(function (b) {
            b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          });
          var scope = document.getElementById(bar.dataset.redraw);
          if (!scope) return;
          var param = bar.dataset.param || "view";
          try {
            sessionStorage.setItem("opt:" + param, btn.dataset.value);
          } catch (e) {
            /* ignore */
          }
          scope.querySelectorAll("img[data-src]").forEach(function (img) {
            var wrap = img.closest(".imgwrap") || img.parentNode;
            wrap.classList.add("busy");
            var url = new URL(img.dataset.src, location.href);
            url.searchParams.set(param, btn.dataset.value);
            var probe = new Image();
            probe.onload = function () {
              img.src = url.toString();
              wrap.classList.remove("busy");
            };
            probe.onerror = function () {
              wrap.classList.remove("busy");
            };
            probe.src = url.toString();
          });
        });
      });
    });
  }

  /* ------------------------------------------------------------ a control inside a panel */
  /* A slider or weight that changes what is computed, not just how it is drawn: the panel asks the
     server for its own new contents rather than reloading the whole page. */
  function initPanelForms(root) {
    root.querySelectorAll("form[data-panel-form]").forEach(function (form) {
      if (form.dataset.wired === "1") return;
      form.dataset.wired = "1";
      var panel = document.getElementById(form.dataset.panelForm);
      var timer = null;

      function send() {
        if (!panel) return;
        var url = new URL(form.getAttribute("action"), location.href);
        new FormData(form).forEach(function (v, k) {
          url.searchParams.set(k, v);
        });
        panel.classList.add("loading");
        fetch(url.toString(), { credentials: "same-origin" })
          .then(function (r) {
            return r.text();
          })
          .then(function (html) {
            panel.innerHTML = html;
            panel.classList.remove("loading");
            initPanelForms(panel);
            initFigBars(panel);
            initTabs(panel);
            drawPlotly(panel);
          })
          .catch(function () {
            panel.classList.remove("loading");
          });
      }

      form.addEventListener("submit", function (e) {
        e.preventDefault();
        send();
      });
      form.addEventListener("input", function (e) {
        var out = form.querySelector("output");
        if (out && e.target.type === "range")
          out.textContent = Number(e.target.value).toLocaleString();
        clearTimeout(timer);
        timer = setTimeout(send, 450);
      });
      form.addEventListener("change", function (e) {
        if (e.target.type === "range") return; /* input already covers it */
        clearTimeout(timer);
        timer = setTimeout(send, 50);
      });
    });
  }

  /* ------------------------------------------------------------ dependent fields */
  /* "Other EPSG code…" reveals a number box; the Map/Section choice hides the controls that only
     belong to a section; a select marked data-reload re-reads the file with the new sheet. */
  function initFields(root) {
    root.querySelectorAll("select[data-crs]").forEach(function (sel) {
      var box = sel.parentNode.querySelector(".crs-epsg");
      if (!box) return;
      function sync() {
        box.hidden = sel.value.indexOf("Other") !== 0;
      }
      sel.addEventListener("change", sync);
      sync();
    });

    root.querySelectorAll("select[data-reload]").forEach(function (sel) {
      sel.addEventListener("change", function () {
        var form = sel.closest("form");
        if (form) form.submit();
      });
    });

    /* A field that only matters when a particular checkbox is ticked. */
    root.querySelectorAll("[data-needs]").forEach(function (el) {
      var form = el.closest("form");
      var box = form && form.querySelector('[name="' + el.dataset.needs + '"]');
      if (!box) {
        el.classList.add("show");
        return;
      }
      function sync() {
        el.classList.toggle("show", box.checked);
      }
      box.addEventListener("change", sync);
      sync();
    });

    root.querySelectorAll("[data-needs-not]").forEach(function (el) {
      var form = el.closest("form");
      var box =
        form && form.querySelector('[name="' + el.dataset.needsNot + '"]');
      if (!box) {
        el.classList.add("show");
        return;
      }
      function sync() {
        el.classList.toggle("show", !box.checked);
      }
      box.addEventListener("change", sync);
      sync();
    });

    root.querySelectorAll("[data-only]").forEach(function (el) {
      var form = el.closest("form");
      if (!form) return;
      var radios = form.querySelectorAll('input[type=radio][name$="layout"]');
      if (!radios.length) return;
      function sync() {
        var chosen = form.querySelector(
          'input[type=radio][name$="layout"]:checked',
        );
        var isSection =
          chosen && chosen.value.toLowerCase().indexOf("section") === 0;
        el.hidden = (el.dataset.only === "section") !== !!isSection;
      }
      radios.forEach(function (r) {
        r.addEventListener("change", sync);
      });
      sync();
    });
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = String(s == null ? "" : s);
    return d.innerHTML;
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTabs(document);
    initLazy(document);
    initFileBoxes(document);
    initFields(document);
    initPanelForms(document);
    initFigBars(document);
    initJob(document);
    drawPlotly(document);
  });
})();
