#ifndef NEUROEDGE_FEATURE_PACKET_H
#define NEUROEDGE_FEATURE_PACKET_H
/* Wire ABI for PRECOMPUTED neural features only. No raw analog acquisition. */
#include <stddef.h>
#include <stdint.h>

#define NE_FEATURE_COUNT 512u
#define NE_PACKET_BYTES 540u
#define NE_PACKET_MAGIC 0x5255454eu /* little-endian bytes: 4e 45 55 52 => "NEUR" */
#define NE_PACKET_VERSION 1u
#define NE_FLAG_BEGIN_TRIAL 0x0001u
#define NE_FLAG_END_TRIAL   0x0002u
#define NE_FLAGS_ALLOWED (NE_FLAG_BEGIN_TRIAL | NE_FLAG_END_TRIAL)

typedef struct {
    uint16_t flags;
    uint32_t trial_id;
    uint32_t bin_index;
    uint64_t timestamp_us;
    int8_t features[NE_FEATURE_COUNT];
} ne_feature_packet;

/* Returns 0 on success. Errors: -1 argument, -2 length, -3 invalid wire. */
int ne_encode_packet(const ne_feature_packet *p, uint8_t *out, size_t n);
int ne_decode_packet(const uint8_t *in, size_t n, ne_feature_packet *out);
uint32_t ne_packet_crc32(const uint8_t *bytes, size_t n);

#endif
