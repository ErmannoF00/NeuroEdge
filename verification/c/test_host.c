#include "neuroedge_host.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t weights,metadata,configured,bias_seen,fault;
    uint32_t in_count,result_count,frames,offset_writes,denied,staged,stages_accepted;
    uint32_t last_bias;
    uint32_t output_word;
    int force_deny, transport_fail;
} mock_t;
static int mock_transfer(void *context,int is_write,uint32_t addr,
                         uint32_t *data,uint32_t timeout) {
    mock_t *m=(mock_t*)context;
    if(!timeout || m->transport_fail) return -1;
    if(is_write) {
        switch(addr) {
        case NEH_WEIGHT_DATA:
            if(!m->configured && m->weights<NEH_WEIGHTS) m->weights++;
            else m->fault=2;
            break;
        case NEH_META_BIAS:
            m->bias_seen=1;m->last_bias=*data;break;
        case NEH_META_SHIFT:
            if(!m->configured && m->bias_seen && m->metadata<NEH_META && *data<=31) {
                m->metadata++;m->bias_seen=0;
            } else m->fault=2;
            break;
        case NEH_SEAL:
            if(*data==1 && m->weights==NEH_WEIGHTS && m->metadata==NEH_META) m->configured=1;
            else m->fault=4;
            break;
        case NEH_INPUT: if(m->configured) m->in_count++;else m->fault=4;break;
        case NEH_OFFSET_CHANNEL: if(*data>=512) m->fault=4;break;
        case NEH_OFFSET_VALUE: if(!m->staged) {m->staged=1;m->stages_accepted++;} else m->fault=4;break;
        case NEH_OFFSET_CTRL:
            if(*data==1 && m->staged) {
                if(m->force_deny) m->denied++;
                else {m->offset_writes++;m->staged=0;}
            } else if(*data==2) m->staged=0;
            else m->fault=4;
            break;
        default: break;
        }
    } else {
        switch(addr) {
        case NEH_ID: *data=NEH_IDENT;break;
        case NEH_STATUS: *data=(m->configured?NEH_CONFIGURED:0) |
                               (m->fault?NEH_FAULT_PRESENT:0) |
                               (m->result_count?NEH_RESULT_READY:0);break;
        case NEH_WEIGHT_PTR: *data=m->weights;break;
        case NEH_META_PTR: *data=m->metadata;break;
        case NEH_FAULT: *data=m->fault;break;
        case NEH_RESULT_FRAME: *data=m->frames;break;
        case NEH_RESULT: *data=m->output_word;m->result_count--;break;
        case NEH_OFFSET_STATUS: *data=m->staged | (!m->staged?2u:0u);break;
        case NEH_STAGES: *data=m->stages_accepted;break;
        case NEH_WRITES: *data=m->offset_writes;break;
        case NEH_REJECTS: *data=m->denied;break;
        default: *data=0;break;
        }
    }
    return 0;
}
int main(void) {
    mock_t mock={0};
    neh_device_t dev={&mock,mock_transfer,100};
    int8_t *w=(int8_t*)calloc(NEH_WEIGHTS,sizeof(int8_t));
    int32_t bias[NEH_META];uint8_t shifts[NEH_META];
    int8_t features[NEH_FEATURES];
    neh_result_t r;
    assert(w);
    for(size_t i=0;i<NEH_META;i++) {bias[i]=(int32_t)i-150;shifts[i]=3;}
    for(size_t i=0;i<NEH_FEATURES;i++) features[i]=(int8_t)(i%128);
    assert(neh_pop_result(&dev,&r)==NEH_ERR_EMPTY);
    assert(neh_configure(&dev,w,bias,shifts)==NEH_OK);
    assert(mock.weights==NEH_WEIGHTS && mock.metadata==NEH_META && mock.configured);
    assert(neh_configure(&dev,w,bias,shifts)==NEH_ERR_STATE);
    assert(neh_send_bin(&dev,features)==NEH_OK);
    assert(mock.in_count==NEH_FEATURES);
    mock.frames=12;mock.output_word=(7u<<16)|(uint16_t)(int16_t)-91;
    mock.result_count=1;
    assert(neh_pop_result(&dev,&r)==NEH_OK);
    assert(r.frame==12 && r.phoneme_class==7 && r.score==-91);
    assert(neh_pop_result(&dev,&r)==NEH_ERR_EMPTY);
    assert(neh_commit_offset(&dev,304,-4)==NEH_OK && mock.offset_writes==1);
    mock.force_deny=1;
    assert(neh_commit_offset(&dev,7,8)==NEH_ERR_DENIED);
    assert(mock.staged==0); // a rejected candidate is always aborted
    mock.force_deny=0;
    mock.staged=1; // stale candidate must never be committed
    assert(neh_commit_offset(&dev,9,12)==NEH_ERR_STATE);
    assert(mock.offset_writes==1);
    assert(neh_abort_offset(&dev)==NEH_OK && !mock.staged);
    assert(neh_commit_offset(&dev,512,3)==NEH_ERR_ARGS);
    mock.transport_fail=1;
    assert(neh_send_bin(&dev,features)==NEH_ERR_BUS);
    free(w);
    puts("NEUROEDGE_HOST_FIRMWARE_CONTRACT_PASS");
    return 0;
}
