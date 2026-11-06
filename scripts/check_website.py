"""Static pages/evidence integrity; excludes browser and accessibility certification."""
from pathlib import Path
import json
import re
from html.parser import HTMLParser

class Inspect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.images: list[dict] = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            if a['id'] in self.ids: raise ValueError(f'duplicate id {a["id"]}')
            self.ids.add(a['id'])
        if tag in ('a','script','link','img'):
            key = 'href' if tag in ('a','link') else 'src'
            if key in a: self.links.append(a[key])
        if tag == 'img': self.images.append(a)


def main():
    root = Path(__file__).resolve().parents[1] / 'website'
    page = (root / 'index.html').read_text()
    evidence = json.loads((root / 'evidence.json').read_text())
    replay = json.loads((root / 'demo/closed_loop_replay.json').read_text())
    inspect = Inspect(); inspect.feed(page)
    assert evidence['verified_software_rtl']['logits_compared'] == 328
    assert evidence['verified_software_rtl']['simulated_latency_cycles'] == 90195
    assert evidence['limitations']['clinical_validation'] is False
    synth = evidence['synthetic_supervised_feedback_replay']
    assert synth['human_feedback'] is False and synth['hardware_adaptation_tested'] is False
    for field, key in [('fixed_logit_l1','fixed_logit_l1'),
                       ('controlled_logit_l1','controlled_logit_l1'),
                       ('accepted_writes','accepted_memory_writes')]:
        assert synth[field] == replay[key], (field,synth[field],replay[key])
    assert synth['sham_logit_l1'] == replay['sham_logit_l1']
    assert synth['sham_accepted_writes'] == replay['sham_accepted_writes']
    assert len(replay['trials_log']) == replay['trials'] == 8
    assert replay['no_hardware_closed_loop_claim'] is True
    assert replay['no_real_data_accuracy_claim'] is True
    assert any(x['src'] == 'assets/system_architecture.svg' for x in inspect.images)
    assert all(x.get('alt') for x in inspect.images)
    for target in inspect.links:
        if target.startswith('#'):
            assert target[1:] in inspect.ids, f'broken anchor {target}'
        elif not re.match(r'^[a-z]+:',target,flags=re.I):
            assert (root / target.split('#')[0]).is_file(), f'missing local asset {target}'
    for name in ['styles.css','calibration.js','replay.js','navigation.js',
                 'assets/system_architecture.png','assets/system_architecture.svg']:
        assert (root/name).is_file(), name
    # No site content should imply that an engineering version number is a product name.
    assert not re.search(r'\bR[2345]\.[0-9]+\b', page)
    # Third-party fonts/analytics violate the offline-first preview contract.
    assert 'fonts.googleapis.com' not in (root/'styles.css').read_text()
    print('STATIC_WEBSITE_EVIDENCE_CHECK_PASS')

if __name__ == '__main__': main()
