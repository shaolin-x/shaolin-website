/*
 * Custom JavaScript for smooth scrolling and other interactive behavior.
 *
 * This script intercepts clicks on navbar links that reference anchors on
 * the same page, scrolling smoothly to the target section while
 * compensating for the fixed navbar height. It also collapses the navbar
 * on small screens after a link is clicked.
 */

document.addEventListener('DOMContentLoaded', function () {
  // Smooth scrolling for anchor links
  const offset = 70; // adjust for fixed navbar height
  const navLinks = document.querySelectorAll('#navbar a.nav-link[href^="#"]');
  navLinks.forEach(function (link) {
    link.addEventListener('click', function (event) {
      const targetId = this.getAttribute('href');
      if (targetId && targetId.startsWith('#')) {
        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          event.preventDefault();
          window.scrollTo({
            top: targetEl.offsetTop - offset,
            behavior: 'smooth'
          });
        }
      }
    });
  });
});