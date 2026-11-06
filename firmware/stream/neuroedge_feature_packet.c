#include "neuroedge_feature_packet.h"
#include <string.h>

static void put16(uint8_t *p, uint16_t x) {
    p[0]=(uint8_t)x; p[1]=(uint8_t)(x >> 8);
}
static void put32(uint8_t *p, uint32_t x) {
    for (unsigned i=0;i<4;i++) p[i]=(uint8_t)(x >> (8*i));
}
static void put64(uint8_t *p, uint64_t x) {
    for (unsigned i=0;i<8;i++) p[i]=(uint8_t)(x >> (8*i));
}
static uint16_t get16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | (uint16_t)((uint16_t)p[1]<<8));
}
static uint32_t get32(const uint8_t *p) {
    uint32_t x=0;
    for (unsigned i=0;i<4;i++) x |= ((uint32_t)p[i] << (8*i));
    return x;
}
static uint64_t get64(const uint8_t *p) {
    uint64_t x=0;
    for (unsigned i=0;i<8;i++) x |= ((uint64_t)p[i] << (8*i));
    return x;
}
uint32_t ne_packet_crc32(const uint8_t *bytes, size_t n) {
    uint32_t x=0xffffffffu;
    if (!bytes && n) return 0;
    for (size_t i=0;i<n;i++) {
        x ^= (uint32_t)bytes[i];
        for (unsigned j=0;j<8;j++)
            x=(x >> 1) ^ ((x & 1u) ? 0xedb88320u : 0u);
    }
    return x ^ 0xffffffffu;
}
int ne_encode_packet(const ne_feature_packet *p, uint8_t *out, size_t n) {
    if (!p || !out) return -1;
    if (n != NE_PACKET_BYTES) return -2;
    if ((p->flags & ~NE_FLAGS_ALLOWED) != 0u) return -3;
    /* Little-endian fields; no C-struct packing/padding assumptions. */
    put32(out, NE_PACKET_MAGIC);
    put16(out+4, NE_PACKET_VERSION);
    put16(out+6, p->flags);
    put32(out+8, p->trial_id);
    put32(out+12, p->bin_index);
    put64(out+16, p->timestamp_us);
    memcpy(out+24, p->features, NE_FEATURE_COUNT);
    put32(out+536, ne_packet_crc32(out,536));
    return 0;
}
int ne_decode_packet(const uint8_t *in, size_t n, ne_feature_packet *out) {
    if (!out || !in) return -1;
    if (n != NE_PACKET_BYTES) return -2;
    if (get32(in) != NE_PACKET_MAGIC || get16(in+4) != NE_PACKET_VERSION ||
        (get16(in+6) & ~NE_FLAGS_ALLOWED) != 0u ||
        get32(in+536) != ne_packet_crc32(in,536)) return -3;
    /* On decode failure, output is not partially updated. */
    ne_feature_packet temp;
    temp.flags=get16(in+6);
    temp.trial_id=get32(in+8);
    temp.bin_index=get32(in+12);
    temp.timestamp_us=get64(in+16);
    memcpy(temp.features, in+24, NE_FEATURE_COUNT);
    *out=temp;
    return 0;
}
