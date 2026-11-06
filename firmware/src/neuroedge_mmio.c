#include "neuroedge_mmio.h"
#include <stddef.h>
/* Register-write shell for a bus adapter; actual platform access layer TBD.
 * Returns negative on invalid inputs / unexpected hardware status.
 * The callback must guarantee completion of a ready/valid transaction.
 */
int ne_configure(void *ctx, ne_write_fn wr, ne_read_fn rd,
                   const int8_t *w, const int32_t *bias,
                   const uint8_t *shift) {
  if(!ctx || !wr || !rd || !w || !bias || !shift) return -1;
  if(rd(ctx,NE_ID)!=NE_IDENT) return -2;
  if(rd(ctx,NE_STATUS)&NE_STATUS_COMMITTED) return -3;
  for(size_t i=0;i<NE_NWEIGHTS;i++) wr(ctx,NE_WEIGHT_DATA,(uint8_t)w[i]);
  for(size_t i=0;i<NE_NMETA;i++) {
    wr(ctx,NE_META_BIAS,(uint32_t)bias[i]);
    wr(ctx,NE_META_SHIFT,(uint32_t)shift[i]);
  }
  wr(ctx,NE_COMMIT,1);
  uint32_t status=rd(ctx,NE_STATUS);
  if(!(status&NE_STATUS_COMMITTED) || (status&NE_STATUS_FAULT)) return -4;
  return 0;
}
