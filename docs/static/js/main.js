/* AdvRole project page interactions */

(function () {
    'use strict';

    /* ---- navbar + scroll progress ---- */
    var navbar = document.getElementById('navbar');
    var progressBar = document.getElementById('progressBar');

    function onScroll() {
        var y = window.scrollY || document.documentElement.scrollTop;
        if (navbar) navbar.classList.toggle('scrolled', y > 24);
        if (progressBar) {
            var h = document.documentElement.scrollHeight - window.innerHeight;
            progressBar.style.width = h > 0 ? (y / h) * 100 + '%' : '0%';
        }
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    /* ---- active nav link ---- */
    var navLinks = Array.prototype.slice.call(document.querySelectorAll('.nav-link'));
    var sections = navLinks
        .map(function (a) { return document.querySelector(a.getAttribute('href')); })
        .filter(Boolean);

    if ('IntersectionObserver' in window && sections.length) {
        var navObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) return;
                var id = entry.target.getAttribute('id');
                navLinks.forEach(function (a) {
                    a.classList.toggle('active', a.getAttribute('href') === '#' + id);
                });
            });
        }, { rootMargin: '-40% 0px -55% 0px' });
        sections.forEach(function (s) { navObserver.observe(s); });
    }

    /* ---- reveal on scroll ---- */
    var revealEls = document.querySelectorAll('.reveal');
    if ('IntersectionObserver' in window) {
        var revealObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.12, rootMargin: '0px 0px -30px 0px' });
        revealEls.forEach(function (el) { revealObserver.observe(el); });
    } else {
        revealEls.forEach(function (el) { el.classList.add('visible'); });
    }

    /* ---- benchmark tabs ---- */
    var tabs = document.querySelectorAll('.tab');
    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            tabs.forEach(function (t) {
                t.classList.remove('active');
                t.setAttribute('aria-selected', 'false');
            });
            tab.classList.add('active');
            tab.setAttribute('aria-selected', 'true');
            document.querySelectorAll('.tab-panel').forEach(function (p) {
                var isActive = p.id === tab.dataset.panel;
                p.classList.toggle('active', isActive);
                p.hidden = !isActive;
            });
        });
    });

    /* ---- bibtex copy ---- */
    function flashBtn(btn) {
        btn.classList.add('copied');
        var label = btn.querySelector('.copy-label');
        var old = label ? label.textContent : '';
        if (label) label.textContent = 'Copied';
        setTimeout(function () {
            btn.classList.remove('copied');
            if (label) label.textContent = old || 'Copy';
        }, 1800);
    }

    var copyBtn = document.getElementById('copyBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', function () {
            var text = document.getElementById('bibtex').textContent;
            navigator.clipboard.writeText(text).then(function () { flashBtn(copyBtn); });
        });
    }

    var quickBtn = document.getElementById('citeQuickBtn');
    if (quickBtn) {
        quickBtn.addEventListener('click', function () {
            var text = document.getElementById('bibtex').textContent;
            navigator.clipboard.writeText(text).then(function () {
                var old = quickBtn.innerHTML;
                quickBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6L9 17l-5-5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg> Copied';
                setTimeout(function () { quickBtn.innerHTML = old; }, 1800);
            });
        });
    }
})();
