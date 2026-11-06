`timescale 1ns/1ps
// NeuroEdge private host register adapter for the VERIFIED speech pipeline.
// Research-specific ready/valid register interface, NOT APB/AHB/AXI.
// One request transfers on posedge clk iff req_valid && req_ready.
// For reads, sample req_rdata on that same accepted edge. RESULT dequeue
// is destructive; read RESULT_FRAME before RESULT for the same FIFO head.
// No DMA, cache coherency, physical memory macros, electrical isolation or
// biological intent authorization is implemented here.
// The pipeline, decoder and transactional offset bank are UNMODIFIED.
module neuroedge_host_adapter #(
    parameter integer INPUT_DEPTH = 4,
    parameter integer OUTPUT_DEPTH = 8
)(
    input  logic clk, rst_n,
    input  logic req_valid, req_write,
    input  logic [7:0] req_addr,
    input  logic [31:0] req_wdata,
    output logic req_ready,
    output logic [31:0] req_rdata,
    output logic irq
);
    localparam integer WEIGHT_COUNT=44224, META_COUNT=361;
    localparam logic [31:0] IDENT = 32'h4e454443; // ASCII NEDC
    localparam logic [7:0]
      REG_ID=8'h00, REG_STATUS=8'h04,
      REG_WPTR=8'h08, REG_WDATA=8'h0c,
      REG_MPTR=8'h10, REG_MBIAS=8'h14,
      REG_MSHIFT=8'h18, REG_SEAL=8'h1c,
      REG_INPUT=8'h20, REG_RESULT=8'h24,
      REG_RESULT_FRAME=8'h28, REG_BINS_DONE=8'h2c,
      REG_OFFSET_CHANNEL=8'h30, REG_OFFSET_VALUE=8'h34,
      REG_OFFSET_CTRL=8'h38, REG_OFFSET_STATUS=8'h3c,
      REG_WRITES=8'h40, REG_REJECTS=8'h44,
      REG_REQUESTS=8'h48, REG_INPUTS_ACCEPTED=8'h4c,
      REG_RESULTS_POPPED=8'h50, REG_IRQ_ENABLE=8'h54,
      REG_IRQ_STATUS=8'h58, REG_IRQ_ACK=8'h5c,
      REG_FAULT=8'h60, REG_FIFO_LEVEL=8'h64,
      REG_LAST_LATENCY=8'h68, REG_STAGES=8'h6c,
      REG_ABORTS=8'h70, REG_VERSION=8'h74;
    localparam logic [31:0]
      FAULT_BAD_REG=32'h1, FAULT_CONFIG=32'h2,
      FAULT_SEQUENCE=32'h4, FAULT_UPDATE_DENIED=32'h8;
    localparam integer IW=(INPUT_DEPTH<=2)?1:$clog2(INPUT_DEPTH);
    localparam integer OW=(OUTPUT_DEPTH<=2)?1:$clog2(OUTPUT_DEPTH);

    logic configured, bias_valid, stage_channel_valid;
    logic [15:0] weight_ptr;
    logic [8:0] meta_ptr, stage_channel;
    logic signed [31:0] pending_bias;
    logic [31:0] faults, irq_enable;
    logic completion_irq_pending;
    logic [31:0] inputs_accepted, results_popped;

    logic signed [7:0] input_fifo [0:INPUT_DEPTH-1];
    logic [IW-1:0] in_head, in_tail;
    logic [IW:0] in_count;
    logic signed [15:0] output_value [0:OUTPUT_DEPTH-1];
    logic [5:0] output_index [0:OUTPUT_DEPTH-1];
    logic [31:0] output_frame [0:OUTPUT_DEPTH-1];
    logic [OW-1:0] out_head, out_tail;
    logic [OW:0] out_count;

    wire cfg_ready, feature_ready, stage_ready;
    wire request_accepted, request_rejected, pending, write_accepted;
    wire [31:0] accepted_requests,rejected_requests,queued_requests;
    wire [31:0] accepted_writes,stages_accepted,aborted_stages;
    wire bin_in_flight,frame_done,busy;
    wire [8:0] feature_index;
    wire [31:0] bin_latency_cycles,bins_completed;
    wire logit_valid;
    wire [5:0] logit_index;
    wire signed [15:0] logit_data;
    wire [31:0] logit_frame;

    wire fire = req_valid && req_ready;
    wire req_w = fire && req_write;
    wire req_r = fire && !req_write;
    wire valid_cfg = !configured && cfg_ready;
    wire do_weight = req_w && req_addr==REG_WDATA &&
                     valid_cfg && weight_ptr<WEIGHT_COUNT;
    wire do_meta = req_w && req_addr==REG_MSHIFT &&
                   valid_cfg && meta_ptr<META_COUNT && bias_valid &&
                   req_wdata<=32;
    wire do_stage = req_w && req_addr==REG_OFFSET_VALUE &&
                    configured && stage_channel_valid && stage_ready &&
                    req_wdata[31:16]=={16{req_wdata[15]}};
    wire ctrl_commit = req_w && req_addr==REG_OFFSET_CTRL &&
                       configured && req_wdata==32'h1;
    wire ctrl_abort = req_w && req_addr==REG_OFFSET_CTRL &&
                      configured && req_wdata==32'h2;
    wire input_push = req_w && req_addr==REG_INPUT && configured && in_count<INPUT_DEPTH;
    wire input_pop = (in_count!=0) && feature_ready;
    wire output_push = logit_valid && (out_count<OUTPUT_DEPTH);
    wire output_pop = req_r && req_addr==REG_RESULT && (out_count!=0);
    logic [31:0] fault_set;
    wire [31:0] fault_clear = (req_w && req_addr==REG_FAULT)?req_wdata:32'd0;
    always_comb begin
        fault_set=0;
        if(req_w) begin
            case(req_addr)
                REG_WDATA: if(!do_weight) fault_set=FAULT_CONFIG;
                REG_MBIAS: if(configured || !cfg_ready || meta_ptr>=META_COUNT)
                                 fault_set=FAULT_CONFIG;
                REG_MSHIFT: if(!do_meta) fault_set=FAULT_CONFIG;
                REG_SEAL: if(configured || weight_ptr!=WEIGHT_COUNT ||
                              meta_ptr!=META_COUNT || bias_valid ||
                              !cfg_ready || req_wdata!=1) fault_set=FAULT_SEQUENCE;
                REG_INPUT: if(!configured) fault_set=FAULT_SEQUENCE;
                REG_OFFSET_CHANNEL: if(!configured || req_wdata>=512)
                                       fault_set=FAULT_SEQUENCE;
                REG_OFFSET_VALUE: if(!do_stage) fault_set=FAULT_SEQUENCE;
                REG_OFFSET_CTRL: if(!ctrl_commit && !ctrl_abort)
                                       fault_set=FAULT_SEQUENCE;
                REG_IRQ_ENABLE,REG_IRQ_ACK,REG_FAULT: ;
                default: fault_set=FAULT_BAD_REG;
            endcase
        end else if(req_r) begin
            case(req_addr)
                REG_ID,REG_STATUS,REG_WPTR,REG_MPTR,REG_RESULT,
                REG_RESULT_FRAME,REG_BINS_DONE,REG_OFFSET_CHANNEL,
                REG_OFFSET_STATUS,REG_WRITES,REG_REJECTS,REG_REQUESTS,
                REG_INPUTS_ACCEPTED,REG_RESULTS_POPPED,REG_IRQ_ENABLE,
                REG_IRQ_STATUS,REG_FAULT,REG_FIFO_LEVEL,REG_LAST_LATENCY,
                REG_STAGES,REG_ABORTS,REG_VERSION: ;
                default: fault_set=FAULT_BAD_REG;
            endcase
        end
    end
    wire [7:0] in_level8 = in_count;
    wire [7:0] out_level8 = out_count;
    wire [31:0] irq_status = {29'd0,completion_irq_pending,(faults!=0),(out_count!=0)};
    assign irq = |(irq_enable[2:0] & irq_status[2:0]);

    function automatic [IW-1:0] input_next(input [IW-1:0] x);
        if(x==INPUT_DEPTH-1) input_next=0; else input_next=x+1'b1;
    endfunction
    function automatic [OW-1:0] output_next(input [OW-1:0] x);
        if(x==OUTPUT_DEPTH-1) output_next=0; else output_next=x+1'b1;
    endfunction

    // Valid accesses which cannot proceed wait; illegal requests transfer and
    // raise sticky fault bits, never silently becoming legal later.
    always_comb begin
        req_ready=1'b0;
        if (req_valid) begin
            req_ready=1'b1;
            if (req_write && (req_addr==REG_WDATA || req_addr==REG_MBIAS ||
                              req_addr==REG_MSHIFT)) begin
                if(!configured && ((req_addr==REG_WDATA && weight_ptr<WEIGHT_COUNT) ||
                     (req_addr!=REG_WDATA && meta_ptr<META_COUNT)))
                    req_ready=cfg_ready;
            end
            if(req_write && req_addr==REG_INPUT && configured)
                req_ready=(in_count<INPUT_DEPTH);
            if(!req_write && req_addr==REG_RESULT)
                req_ready=(out_count!=0);
        end
    end

    always_comb begin
        req_rdata=32'd0;
        case(req_addr)
            REG_ID: req_rdata=IDENT;
            REG_STATUS: req_rdata={24'd0,irq,(in_count==INPUT_DEPTH),
                                   bin_in_flight,pending,(faults!=0),
                                   (out_count!=0),busy,configured};
            REG_WPTR: req_rdata={16'd0,weight_ptr};
            REG_MPTR: req_rdata={23'd0,meta_ptr};
            REG_RESULT: if(out_count!=0)
                req_rdata={10'd0,output_index[out_head],output_value[out_head]};
            REG_RESULT_FRAME: if(out_count!=0) req_rdata=output_frame[out_head];
            REG_BINS_DONE: req_rdata=bins_completed;
            REG_OFFSET_CHANNEL: req_rdata={23'd0,stage_channel};
            REG_OFFSET_STATUS: req_rdata={25'd0,busy,bin_in_flight,
                                          request_rejected,request_accepted,
                                          write_accepted,stage_ready,pending};
            REG_WRITES: req_rdata=accepted_writes;
            REG_REJECTS: req_rdata=rejected_requests;
            REG_REQUESTS: req_rdata=accepted_requests;
            REG_INPUTS_ACCEPTED: req_rdata=inputs_accepted;
            REG_RESULTS_POPPED: req_rdata=results_popped;
            REG_IRQ_ENABLE: req_rdata=irq_enable;
            REG_IRQ_STATUS: req_rdata=irq_status;
            REG_FAULT: req_rdata=faults;
            REG_FIFO_LEVEL: req_rdata={16'd0,out_level8,in_level8};
            REG_LAST_LATENCY: req_rdata=bin_latency_cycles;
            REG_STAGES: req_rdata=stages_accepted;
            REG_ABORTS: req_rdata=aborted_stages;
            REG_VERSION: req_rdata=32'h00010000;
            default: req_rdata=0;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            configured<=0; bias_valid<=0; stage_channel_valid<=0;
            weight_ptr<=0; meta_ptr<=0; stage_channel<=0;
            pending_bias<=0; faults<=0; irq_enable<=0;
            completion_irq_pending<=0;
            inputs_accepted<=0; results_popped<=0;
            in_head<=0;in_tail<=0;in_count<=0;
            out_head<=0;out_tail<=0;out_count<=0;
        end else begin
            // A newly completed bin wins over a simultaneous IRQ ACK.
            if (req_w && req_addr==REG_IRQ_ACK && req_wdata[2])
                completion_irq_pending<=0;
            if (frame_done) completion_irq_pending<=1;
            // New faults dominate a simultaneous W1C acknowledgement.
            faults <= (faults & ~fault_clear) | fault_set |
                       (request_rejected ? FAULT_UPDATE_DENIED : 32'd0);
            if (input_push) begin
                input_fifo[in_tail]<=req_wdata[7:0];
                in_tail<=input_next(in_tail);
                inputs_accepted<=inputs_accepted+1'b1;
            end
            if (input_pop) in_head<=input_next(in_head);
            case ({input_push,input_pop})
                2'b10: in_count<=in_count+1'b1;
                2'b01: in_count<=in_count-1'b1;
                default: ;
            endcase
            if(output_push) begin
                output_value[out_tail]<=logit_data;
                output_index[out_tail]<=logit_index;
                output_frame[out_tail]<=logit_frame;
                out_tail<=output_next(out_tail);
            end
            if(output_pop) begin
                out_head<=output_next(out_head);
                results_popped<=results_popped+1'b1;
            end
            case ({output_push,output_pop})
                2'b10: out_count<=out_count+1'b1;
                2'b01: out_count<=out_count-1'b1;
                default: ;
            endcase
            if(req_w) begin
                case(req_addr)
                    REG_WDATA: if(do_weight) weight_ptr<=weight_ptr+1'b1;
                    REG_MBIAS: if(!configured && cfg_ready && meta_ptr<META_COUNT) begin
                         pending_bias<=$signed(req_wdata);bias_valid<=1'b1;
                    end
                    REG_MSHIFT: if(do_meta) begin
                         meta_ptr<=meta_ptr+1'b1;bias_valid<=1'b0;
                    end
                    REG_SEAL: if(!configured && weight_ptr==WEIGHT_COUNT &&
                                  meta_ptr==META_COUNT && !bias_valid &&
                                  cfg_ready && req_wdata==1)
                                  configured<=1'b1;

                    REG_INPUT: ;
                    REG_OFFSET_CHANNEL: if(configured && req_wdata<512) begin
                         stage_channel<=req_wdata[8:0];stage_channel_valid<=1;
                    end
                    REG_OFFSET_VALUE: if(do_stage) stage_channel_valid<=0;

                    REG_OFFSET_CTRL: ;
                    REG_IRQ_ENABLE: irq_enable<={29'd0,req_wdata[2:0]};
                    REG_IRQ_ACK: ;
                    REG_FAULT: ;
                    default: ;
                endcase
            end
        end
    end

    neuroedge_speech_pipeline u_pipeline (
        .clk(clk),.rst_n(rst_n),
        .cfg_weight_we(do_weight),.cfg_weight_addr(weight_ptr),
        .cfg_weight_data(req_wdata[7:0]),
        .cfg_meta_we(do_meta),.cfg_meta_addr(meta_ptr),
        .cfg_meta_bias(pending_bias),.cfg_meta_shift(req_wdata[4:0]),
        .cfg_ready(cfg_ready),
        .feature_valid(in_count!=0),.feature_ready(feature_ready),
        .feature_data(input_fifo[in_head]),
        .stage_valid(do_stage),.stage_ready(stage_ready),
        .stage_channel(stage_channel),.stage_value(req_wdata[15:0]),
        .abort_stage(ctrl_abort),.authorize_update(ctrl_commit),
        .request_accepted(request_accepted),.request_rejected(request_rejected),
        .accepted_requests(accepted_requests),.rejected_requests(rejected_requests),
        .queued_requests(queued_requests),.pending(pending),
        .write_accepted(write_accepted),.accepted_writes(accepted_writes),
        .stages_accepted(stages_accepted),.aborted_stages(aborted_stages),
        .bin_in_flight(bin_in_flight),.feature_index(feature_index),
        .bin_latency_cycles(bin_latency_cycles),.bins_completed(bins_completed),
        .logit_valid(logit_valid),.logit_ready(out_count<OUTPUT_DEPTH),
        .logit_index(logit_index),.logit_data(logit_data),
        .logit_frame(logit_frame),.frame_done(frame_done),.busy(busy)
    );
    initial begin
        if (INPUT_DEPTH<2 || OUTPUT_DEPTH<2 || INPUT_DEPTH>255 || OUTPUT_DEPTH>255)
            $fatal(1,"NEUROEDGE_FIFO_DEPTH_OUT_OF_RANGE");
    end
endmodule
