#include "neuroedge_host.h"

int neh_read(neh_device_t *dev, uint32_t addr, uint32_t *data) {
    if (!dev || !dev->transfer || !data || !dev->wait_cycles) return NEH_ERR_ARGS;
    return dev->transfer(dev->context,0,addr,data,dev->wait_cycles) ? NEH_ERR_BUS : NEH_OK;
}
int neh_write(neh_device_t *dev, uint32_t addr, uint32_t data) {
    if (!dev || !dev->transfer || !dev->wait_cycles) return NEH_ERR_ARGS;
    return dev->transfer(dev->context,1,addr,&data,dev->wait_cycles) ? NEH_ERR_BUS : NEH_OK;
}
static int check_faults(neh_device_t *dev) {
    uint32_t fault=0;
    int rc=neh_read(dev,NEH_FAULT,&fault);
    if(rc) return rc;
    return fault ? NEH_ERR_FAULT : NEH_OK;
}
int neh_configure(neh_device_t *dev, const int8_t *weights,
                  const int32_t *biases, const uint8_t *shifts) {
    uint32_t v=0;
    int rc;
    if(!weights || !biases || !shifts) return NEH_ERR_ARGS;
    for(size_t i=0;i<NEH_META;i++) if(shifts[i]>31) return NEH_ERR_ARGS;
    if((rc=neh_read(dev,NEH_ID,&v))) return rc;
    if(v!=NEH_IDENT) return NEH_ERR_STATE;
    if((rc=neh_read(dev,NEH_STATUS,&v))) return rc;
    if(v & NEH_CONFIGURED) return NEH_ERR_STATE;
    if((rc=check_faults(dev))) return rc;
    for(size_t i=0;i<NEH_WEIGHTS;i++) {
        if((rc=neh_write(dev,NEH_WEIGHT_DATA,(uint8_t)weights[i]))) return rc;
    }
    if((rc=neh_read(dev,NEH_WEIGHT_PTR,&v))) return rc;
    if(v!=NEH_WEIGHTS) return NEH_ERR_STATE;
    for(size_t i=0;i<NEH_META;i++) {
        if((rc=neh_write(dev,NEH_META_BIAS,(uint32_t)biases[i]))) return rc;
        if((rc=neh_write(dev,NEH_META_SHIFT,shifts[i]))) return rc;
    }
    if((rc=neh_read(dev,NEH_META_PTR,&v))) return rc;
    if(v!=NEH_META) return NEH_ERR_STATE;
    if((rc=neh_write(dev,NEH_SEAL,1))) return rc;
    if((rc=neh_read(dev,NEH_STATUS,&v))) return rc;
    if(!(v&NEH_CONFIGURED)) return NEH_ERR_STATE;
    return check_faults(dev);
}
int neh_send_bin(neh_device_t *dev, const int8_t features[NEH_FEATURES]) {
    int rc; uint32_t v=0;
    if(!features) return NEH_ERR_ARGS;
    if((rc=neh_read(dev,NEH_STATUS,&v))) return rc;
    if(!(v&NEH_CONFIGURED) || (v&NEH_FAULT_PRESENT)) return NEH_ERR_STATE;
    for(size_t i=0;i<NEH_FEATURES;i++)
        if((rc=neh_write(dev,NEH_INPUT,(uint8_t)features[i]))) return rc;
    return check_faults(dev);
}
int neh_pop_result(neh_device_t *dev, neh_result_t *result) {
    int rc; uint32_t status=0,frame=0,word=0;
    if(!result) return NEH_ERR_ARGS;
    if((rc=neh_read(dev,NEH_STATUS,&status))) return rc;
    if(!(status&NEH_RESULT_READY)) return NEH_ERR_EMPTY;
    /* The FIFO head cannot be consumed by anyone else using this
       single-owner bus contract. Read frame BEFORE destructive result. */
    if((rc=neh_read(dev,NEH_RESULT_FRAME,&frame))) return rc;
    if((rc=neh_read(dev,NEH_RESULT,&word))) return rc;
    result->frame=frame;
    result->phoneme_class=(uint8_t)((word>>16)&63u);
    {
        const uint32_t raw=word&65535u;
        result->score=(int16_t)((raw&32768u)?(int32_t)raw-65536:(int32_t)raw);
    }
    return NEH_OK;
}
int neh_commit_offset(neh_device_t *dev, uint16_t channel, int16_t offset) {
    uint32_t before=0, denied=0, after=0, stages=0, v=0;
    int rc;
    if(channel>=NEH_FEATURES) return NEH_ERR_ARGS;
    if((rc=check_faults(dev))) return rc;
    if((rc=neh_read(dev,NEH_OFFSET_STATUS,&v))) return rc;
    /* Never mistake an older pending candidate for the one just submitted. */
    if((v&1u) || !(v&2u)) return NEH_ERR_STATE;
    if((rc=neh_read(dev,NEH_STAGES,&stages))) return rc;
    if((rc=neh_read(dev,NEH_WRITES,&before))) return rc;
    if((rc=neh_read(dev,NEH_REJECTS,&denied))) return rc;
    if((rc=neh_write(dev,NEH_OFFSET_CHANNEL,channel))) return rc;
    if((rc=neh_write(dev,NEH_OFFSET_VALUE,(uint32_t)(int32_t)offset))) return rc;
    if((rc=neh_read(dev,NEH_STAGES,&v))) return rc;
    if(v!=stages+1u) return NEH_ERR_STATE;
    if((rc=neh_read(dev,NEH_OFFSET_STATUS,&v))) return rc;
    if(!(v&1u)) return NEH_ERR_STATE;
    if((rc=check_faults(dev))) return rc;
    if((rc=neh_write(dev,NEH_OFFSET_CTRL,1))) return rc;
    if((rc=neh_read(dev,NEH_WRITES,&after))) return rc;
    if(after==before+1u) return NEH_OK;
    if((rc=neh_read(dev,NEH_REJECTS,&v))) return rc;
    /* Explicitly discard a rejected proposal. The hardware itself never
       queues it for later, but the one-entry staging slot remains pending. */
    if(v!=denied) {
        (void)neh_abort_offset(dev);
        return NEH_ERR_DENIED;
    }
    (void)neh_abort_offset(dev);
    return NEH_ERR_STATE;
}
int neh_abort_offset(neh_device_t *dev) {
    return neh_write(dev,NEH_OFFSET_CTRL,2u);
}
