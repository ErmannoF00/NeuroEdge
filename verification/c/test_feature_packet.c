#include "../../firmware/stream/neuroedge_feature_packet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *s) {fprintf(stderr,"FEATURE_PACKET_C_FAIL %s\n",s);exit(1);}
int main(int argc, char **argv) {
    uint8_t encoded[NE_PACKET_BYTES];
    ne_feature_packet p;
    if (argc!=2) die("expected Python-generated fixture path");
    FILE *f=fopen(argv[1],"rb");
    if (!f)die("open");
    if (fread(encoded,1,sizeof(encoded),f)!=sizeof(encoded))die("read");
    if (fgetc(f)!=EOF)die("extra bytes");
    if (fclose(f))die("close");
    if (ne_decode_packet(encoded,sizeof(encoded),&p))die("decode Python packet");
    if (p.trial_id!=7u || p.bin_index!=3u || p.timestamp_us!=123456789ull || p.flags!=0)
        die("header mismatch");
    for (unsigned i=0;i<NE_FEATURE_COUNT;i++) {
        int8_t want=(int8_t)((int)(((i*37u+11u)%256u))-128);
        if (p.features[i]!=want)die("signed feature mismatch");
    }
    uint8_t rebuilt[NE_PACKET_BYTES];
    if (ne_encode_packet(&p,rebuilt,sizeof(rebuilt)))die("reencode");
    if (memcmp(encoded,rebuilt,NE_PACKET_BYTES))die("cross-language bytes differ");
    uint8_t corrupt[NE_PACKET_BYTES];memcpy(corrupt,encoded,sizeof(corrupt));
    corrupt[31]^=1;
    ne_feature_packet untouched=p;
    if (ne_decode_packet(corrupt,sizeof(corrupt),&p)!=-3)die("CRC must reject corruption");
    if (untouched.flags!=p.flags || untouched.trial_id!=p.trial_id ||
        untouched.bin_index!=p.bin_index || untouched.timestamp_us!=p.timestamp_us ||
        memcmp(untouched.features,p.features,NE_FEATURE_COUNT))
        die("decode modified state on error");
    if (ne_decode_packet(encoded,NE_PACKET_BYTES-1u,&p)!=-2)die("length must reject");
    p.flags=0x4u;
    if (ne_encode_packet(&p,rebuilt,sizeof(rebuilt))!=-3)die("reserved flags must reject");
    puts("NEUROEDGE_PYTHON_C_FEATURE_PACKET_PASS bytes=540 features=512 crc=PASS");
    return 0;
}
