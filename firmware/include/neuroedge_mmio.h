#ifndef NEUROEDGE_MMIO_H
#define NEUROEDGE_MMIO_H
#include <stdint.h>
/* Experimental MMIO register contract. An adapter to X-HEEP's actual
 * interconnect and hardware base-address selection is NOT implemented.
 * Transfer only when valid && ready; busy-wait here is an abstract host
 * driver, not production power-aware firmware. Little endian.
 */
#define NE_ID           0x00u
#define NE_STATUS       0x04u
#define NE_WEIGHT_PTR   0x08u
#define NE_WEIGHT_DATA  0x0cu
#define NE_META_PTR     0x10u
#define NE_META_BIAS    0x14u
#define NE_META_SHIFT   0x18u
#define NE_COMMIT       0x1cu
#define NE_INPUT        0x20u
#define NE_RESULT       0x24u
#define NE_FRAME        0x28u
#define NE_COMPLETED    0x2cu
#define NE_STATUS_COMMITTED  (1u<<0)
#define NE_STATUS_BUSY       (1u<<1)
#define NE_STATUS_RESULT     (1u<<2)
#define NE_STATUS_FAULT      (1u<<3)
#define NE_IDENT      0x4e453139u
#define NE_NWEIGHTS   44224u
#define NE_NMETA      361u
#define NE_CHANNELS   512u
#define NE_CLASSES    41u
#define NE_SCORE(word) ((int16_t)((word)&0xffffu))
#define NE_CLASS(word) (((word)>>16)&0x3fu)
typedef void (*ne_write_fn)(void *, uint32_t, uint32_t);
typedef uint32_t (*ne_read_fn)(void *, uint32_t);
int ne_configure(void *ctx, ne_write_fn wr, ne_read_fn rd,
                   const int8_t *w, const int32_t *bias,
                   const uint8_t *shift);
#endif
