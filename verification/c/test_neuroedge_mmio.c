#include "neuroedge_mmio.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct {
  uint32_t wc,mc,mb_count,commit;
  uint32_t bias;
  uint32_t faults;
} fake_hw;
static void write_fn(void *p,uint32_t addr,uint32_t val) {
  fake_hw *h=(fake_hw *)p;
  switch(addr) {
    case NE_WEIGHT_DATA: h->wc++; break;
    case NE_META_BIAS: h->bias=val;h->mb_count++;break;
    case NE_META_SHIFT: h->mc++;break;
    case NE_COMMIT: h->commit=val;break;
    default: h->faults++;
  }
}
static uint32_t read_fn(void *p,uint32_t addr) {
  fake_hw *h=(fake_hw *)p;
  if(addr==NE_ID) return NE_IDENT;
  if(addr==NE_STATUS) return h->commit ? NE_STATUS_COMMITTED:0;
  return 0;
}
int main(void) {
  static int8_t weights[NE_NWEIGHTS];
  static int32_t bias[NE_NMETA];
  static uint8_t shift[NE_NMETA];
  fake_hw h={0};
  for(uint32_t i=0;i<NE_NWEIGHTS;i++) weights[i]=(int8_t)(i%127);
  for(uint32_t i=0;i<NE_NMETA;i++) {bias[i]=(int32_t)i;shift[i]=(uint8_t)(i%16);}
  assert(ne_configure(NULL,write_fn,read_fn,weights,bias,shift)==-1);
  assert(ne_configure(&h,write_fn,read_fn,weights,bias,shift)==0);
  assert(h.wc==NE_NWEIGHTS);
  assert(h.mb_count==NE_NMETA);
  assert(h.mc==NE_NMETA);
  assert(h.faults==0 && h.commit==1);
  assert(ne_configure(&h,write_fn,read_fn,weights,bias,shift)==-3);
  assert(NE_SCORE(0x0001ffffu)==-1);
  assert(NE_CLASS(0x0001ffffu)==1);
  puts("NEUROEDGE_HOST_FIRMWARE_TEST_PASS");
  return 0;
}
