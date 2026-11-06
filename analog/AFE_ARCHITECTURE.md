# Neural analog front-end architecture

The first silicon milestone should keep the AFE modular. A practical research path is:

```text
electrode / external source
        |
        v
ESD + low-leakage protection
        |
        v
AC-coupled or DC-servo LNA
        |
        v
PGA / anti-alias filter
        |
        v
SAR or delta-sigma ADC
        |
        v
sample FIFO / digital preprocessing
```

For non-invasive EEG, start with an external validated AFE/ADC board and feed recorded samples into the ASIC. A custom neural-recording AFE is a second mixed-signal tapeout problem, not something to hide inside the digital accelerator schedule.

## Schematic deliverables for an AMS version

- electrode equivalent model (`Rs`, `Cdl`, `Rct`, half-cell offset)
- input protection and leakage budget
- chopper or low-flicker LNA
- programmable gain
- DC-servo / baseline restoration
- reference/bias generation
- ADC + clocking
- calibration/test injection
- isolation of analog and digital supplies/grounds at system level

Each block needs AC, transient, noise, PVT, Monte-Carlo, loop-stability and extracted-post-layout verification as applicable.
