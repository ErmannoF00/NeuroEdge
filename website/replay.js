(() => {
  'use strict';
  const next = document.getElementById('replay-next');
  if (!next) return;
  const reset = document.getElementById('replay-reset');
  const view = id => document.getElementById(id);
  let trace = null;
  let index = 0;
  function render() {
    if (!trace) return;
    const trial = trace.trials_log[index];
    view('replay-trial').textContent = `${index} / ${trace.trials_log.length - 1}`;
    view('replay-fixed').textContent = trial.baseline_logit_l1;
    view('replay-controlled').textContent = trial.controlled_logit_l1;
    view('replay-offset').textContent = trial.active_offset_q4 >= 0
      ? `+${trial.active_offset_q4}` : String(trial.active_offset_q4);
    const max = Math.max(1, ...trace.trials_log.map(t => t.baseline_logit_l1),
      ...trace.trials_log.map(t => t.controlled_logit_l1),
      ...trace.trials_log.map(t => t.sham_logit_l1));
    view('replay-fixed-bar').style.width = `${trial.baseline_logit_l1 / max * 100}%`;
    view('replay-sham-bar').style.width = `${trial.sham_logit_l1 / max * 100}%`;
    view('replay-controlled-bar').style.width = `${trial.controlled_logit_l1 / max * 100}%`;
    view('replay-decision').textContent = trial.accepted
      ? `Before trial ${index}: ${trial.decision}; Δoffset = ${trial.proposal_delta_q4} Q4.`
      : `Trial ${index}: ${trial.decision}; no committed write.`;
    next.disabled = index >= trace.trials_log.length - 1;
  }
  next.addEventListener('click', () => { if (trace && index < trace.trials_log.length - 1) { index++; render(); } });
  reset.addEventListener('click', () => { index = 0; render(); });
  fetch('demo/closed_loop_replay.json', { cache: 'no-store' })
    .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
    .then(data => {
      if (!Array.isArray(data.trials_log) || data.trials_log.length !== data.trials ||
          data.no_supervised_target_for_sham !== true ||
          data.no_real_data_accuracy_claim !== true || data.no_hardware_closed_loop_claim !== true) {
        throw new Error('Incompatible synthetic replay evidence');
      }
      trace = data;
      render();
    })
    .catch(() => {
      next.disabled = true;
      view('replay-decision').textContent = 'Synthetic evidence unavailable. Run the site through the local HTTP server.';
    });
})();
