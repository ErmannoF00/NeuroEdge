#ifndef NEUROEDGE_FULL_DECODER_CONTRACT_H
#define NEUROEDGE_FULL_DECODER_CONTRACT_H
/* existing protocol CONTRACT (not an instantiated RISC-V peripheral).
 * All integers little-endian when used in a future memory-mapped wrapper.
 * The core currently exposes a byte-stream, valid/ready and config ports.
 * Do not treat these addresses as silicon MMIO until a bus wrapper exists.
 */
#include <stdint.h>
#define NE_INPUT_CHANNELS       512u
#define NE_HIDDEN_CHANNELS       64u
#define NE_PHONEME_CLASSES      41u
#define NE_OUTPUT_STRIDE_BINS    4u
#define NE_WEIGHT_BYTES      44224u
#define NE_META_ENTRIES        361u
#define NE_INPUT_SCALE_DEN      16u
#define NE_BIAS_SCALE_DEN       16u
#define NE_WEIGHT_BASE_PROJECT     0u
#define NE_WEIGHT_BASE_DW1    32768u
#define NE_WEIGHT_BASE_PW1    33088u
#define NE_WEIGHT_BASE_DW2    37184u
#define NE_WEIGHT_BASE_PW2    37504u
#define NE_WEIGHT_BASE_HEAD   41600u
#define NE_META_BASE_PROJECT     0u
#define NE_META_BASE_DW1        64u
#define NE_META_BASE_PW1       128u
#define NE_META_BASE_DW2       192u
#define NE_META_BASE_PW2       256u
#define NE_META_BASE_HEAD      320u
#endif
