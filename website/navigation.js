(() => {
  'use strict';
  const button = document.querySelector('.menu-toggle');
  const menu = document.getElementById('primary-menu');
  if (!button || !menu) return;
  function toggle(open) {
    button.setAttribute('aria-expanded', open ? 'true' : 'false');
    button.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    menu.dataset.open = open ? 'true' : 'false';
  }
  button.addEventListener('click', () => toggle(button.getAttribute('aria-expanded') !== 'true'));
  menu.addEventListener('click', event => {
    if (event.target.closest('a')) toggle(false);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') toggle(false);
  });
})();
