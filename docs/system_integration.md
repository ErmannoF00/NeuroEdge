# System integration: precise implementation boundaries

This document is the current **source-of-truth interface map**. It is not a clinical product specification. The verified hardware path, the connected software host path, and the prospective analog/CIM components must not be conflated.

## End-to-end composition

```text
Recorded attempted-speech HDF5           Future electrode/AFE/ADC hardware
             │                                      │ (NOT BUILT)
     Train-only normalization                Feature extraction
             │                                      │
      Q4 signed INT8 [512] <────────────────────────┘
             │
    540-byte CRC-protected feature frame (C / Python; HOST PROTOTYPE)
             │ trial index / bin index / timestamp / begin / end
             ▼
     WireStreamReceiver (strict sequence, abort/recovery)
             │
             ▼
     ConnectedStream (frozen trial ownership)
             │   512×INT16 transactional offset bank
             ▼
     Signed INT8 saturating calibration
             │
             ▼
     Causal decoder, 44,224 signed INT8 weights
             │   baseline / logically banked SOFTWARE projection selectable
             ▼
     41 signed INT16 CTC logits / 4 input bins
             │
       CTC collapse / task evaluation
             │
      Explicit independent output arming ─────► optional phoneme release
             │
   (completed-trial task labels available ONLY after trial ends)
             ▼
     Past-trial bounded candidate search
             │
     staged authorization / version gate
             └────────────► next trial's offset memory (never mid-bin)

INDEPENDENT HARDWARE REFERENCE:
  rtl/soc/neuroedge_speech_pipeline.sv
      feature ready/valid -> registered INT16 correction ->
      synchronous speech decoder -> 41 output logits

EXPERIMENTAL HARDWARE:
  rtl/memory/neuroedge_near_memory_dot.sv
      independently tested single signed-INT8 near-memory dot engine
      NOT A 64-BANK INTEGRATED PROJECTOR AND NOT CIM
```

## Exact data / clock contracts

| Interface | Contract | Source | State |
|---|---|---|---|
| Recorded feature | 512 signed INT8 Q4 features every nominal 20 ms bin | `neuroedge/speech/dataset.py`, `neuroedge/speech/fixed_decoder.py` | Offline, recorded |
| Wire format | Fixed 540-byte little-endian structure including CRC32 and trial metadata | `neuroedge/system/packet_wire.py`, `firmware/stream/` | Byte-level tested, no physical transport |
| Trial receiver | First packet has BEGIN, last END; sequential timestamps, contiguous indexes, explicit abort | `neuroedge/system/wire_receiver.py` | Python tests |
| Offset bank | 512 signed INT16 values, same Q4 units as input, safe boundary authorization | `rtl/control/neuroedge_offset_bank.sv`, `neuroedge/speech/offset_reference.py` | RTL / Python simulation verified |
| Calibration | `sat_int8(int17(raw) + int17(offset))`; registered memory read | `rtl/soc/neuroedge_offset_stream_bridge.sv` | RTL verified |
| Decoder | 512→64 projection; 2 depthwise/pointwise stages; 64→41 CTC head | `rtl/speech/sync/neuroedge_speech_decoder.sv` | RTL verified |
| Output | 41 signed INT16 scalars per CTC timestep, one timestep every four input bins | `neuroedge/speech/fixed_decoder.py` | Exact Python/RTL golden agreement |
| Output permission | Separate externally armed decision, before trial | `neuroedge/system/streaming.py` | Software only; not a neural intention detector |
| Feedback | Post-trial labels; candidate only for later trials; max bounded step and abs offset | `neuroedge/control/feedback_policy.py` | Software replay only |

### Time-domain distinction

Input timestamps are **virtual replay timestamps**. `wall_decode_ms` records CPU time for software execution, but is not equivalent to a physical acquisition-to-audio end-to-end latency. `90195` simulation cycles / assumed 25 MHz = 3.6078 ms; it is not a measured post-layout clock or silicon response.

### Security and agency

CRC32 detects accidental corruption. It does not authenticate a neural source or user, encrypt traffic, establish consent, or protect a network connection. External trial arming is a software interface, not evidence of a classifier distinguishing private thoughts from a voluntary communication command. Default output is suppressed. Clinical or stimulation pathways do not exist in this repository.

## Next integration that must be built, rather than drawn

The main outstanding silicon-related interface is a **single host-to-RTL command plane** for writing parameters, reporting counters and faults, conveying explicit output permission, and authorizing prospective calibration proposals. In a future implementation it should have a versioned register map, reset state, configuration readback, input/output FIFOs, interrupt semantics, and watchdog/deadline behavior. Do not treat the older independent MMIO demonstrator as currently wired to the speech pipeline.

A full 512×64 parallel digital projector must own 512 input activations, schedule 64 weight rows, perform signed accumulation, apply exactly the existing per-row requantization, and reproduce 64 projected activations and all 328 synthetic test logits. The current one-row tile is **only** an independently tested unit.

## What remains outside this system

- Raw electrode voltage acquisition, op-amp/ADC performance, isolation, safety and calibration.
- Real-time physical link from an acquisition board to the host/RTL.
- Human-confirmed inner-speech communication intent.
- Text/language model, text-to-speech, actual speaker and adaptive auditory feedback.
- ASIC synthesis with identified SRAM/CIM macros, physical timing and measured power.
- Evidence of calibration improvement on held-out neural trials.

These boundaries are essential for any future public README, technical poster, IP disclosure or demonstration.
