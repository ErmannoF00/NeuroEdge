#ifndef NEUROEDGE_HOST_H
#define NEUROEDGE_HOST_H
#include <stdint.h>
#include <stddef.h>
/* Research-ready/valid 32-bit register endpoint. Caller must implement the
 * transfer callback using an actual bus bridge. No direct RISC-V MMIO is
 * claimed until an interconnect mapping is supplied.
 * A successful read completes exactly one handshake, including RESULT pop.
 */
#define NEH_ID             0x00u
#define NEH_STATUS         0x04u
#define NEH_WEIGHT_PTR     0x08u
#define NEH_WEIGHT_DATA    0x0cu
#define NEH_META_PTR       0x10u
#define NEH_META_BIAS      0x14u
#define NEH_META_SHIFT     0x18u
#define NEH_SEAL           0x1cu
#define NEH_INPUT          0x20u
#define NEH_RESULT         0x24u
#define NEH_RESULT_FRAME   0x28u
#define NEH_BINS_DONE      0x2cu
#define NEH_OFFSET_CHANNEL 0x30u
#define NEH_OFFSET_VALUE   0x34u
#define NEH_OFFSET_CTRL    0x38u
#define NEH_OFFSET_STATUS  0x3cu
#define NEH_WRITES         0x40u
#define NEH_REJECTS        0x44u
#define NEH_REQUESTS       0x48u
#define NEH_INPUTS_ACCEPTED 0x4cu
#define NEH_RESULTS_POPPED 0x50u
#define NEH_IRQ_ENABLE     0x54u
#define NEH_IRQ_STATUS     0x58u
#define NEH_IRQ_ACK        0x5cu
#define NEH_FAULT          0x60u
#define NEH_FIFO_LEVEL     0x64u
#define NEH_LAST_LATENCY   0x68u
#define NEH_STAGES         0x6cu
#define NEH_ABORTS         0x70u
#define NEH_VERSION        0x74u
#define NEH_IDENT         0x4e454443u
#define NEH_CONFIGURED     (1u<<0)
#define NEH_BUSY           (1u<<1)
#define NEH_RESULT_READY   (1u<<2)
#define NEH_FAULT_PRESENT  (1u<<3)
#define NEH_PENDING        (1u<<4)
#define NEH_BIN_IN_FLIGHT  (1u<<5)
#define NEH_IRQ_RESULT     (1u<<0)
#define NEH_IRQ_FAULT      (1u<<1)
#define NEH_IRQ_COMPLETED  (1u<<2)
#define NEH_WEIGHTS        44224u
#define NEH_META           361u
#define NEH_FEATURES       512u
#define NEH_CLASSES        41u

typedef int (*neh_transfer_t)(void *context, int is_write, uint32_t addr,
                              uint32_t *data, uint32_t max_wait_cycles);
typedef struct { void *context; neh_transfer_t transfer; uint32_t wait_cycles; } neh_device_t;
typedef struct { uint32_t frame; uint8_t phoneme_class; int16_t score; } neh_result_t;

enum { NEH_OK=0, NEH_ERR_ARGS=-1, NEH_ERR_BUS=-2,
       NEH_ERR_STATE=-3, NEH_ERR_FAULT=-4, NEH_ERR_EMPTY=-5,
       NEH_ERR_DENIED=-6 };
int neh_read(neh_device_t *dev, uint32_t addr, uint32_t *data);
int neh_write(neh_device_t *dev, uint32_t addr, uint32_t data);
int neh_configure(neh_device_t *dev, const int8_t *weights,
                  const int32_t *biases, const uint8_t *shifts);
int neh_send_bin(neh_device_t *dev, const int8_t features[NEH_FEATURES]);
/* Returns NEH_ERR_EMPTY without accessing destructive RESULT on empty FIFO. */
int neh_pop_result(neh_device_t *dev, neh_result_t *result);
/* Stages absolute INT16 offset; results indicate hardware authorization.
 * POLICY CHECKS (lagged evidence, bounds) must occur in the trusted host. */
int neh_commit_offset(neh_device_t *dev, uint16_t channel, int16_t offset);
int neh_abort_offset(neh_device_t *dev);
#endif
