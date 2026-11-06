`timescale 1ns/1ps
// NeuroEdge existing research MMIO boundary, NOT yet APB/AHB/AXI or a finalized SoC.
// Host drives request/address/write/data and holds until req_ready.
// A request transfers on any rising edge where req_valid && req_ready.
// Multiword parameter loading is locked by explicit COMMIT, before inference.
// A RESULT read is a DESTRUCTIVE dequeue with stable result data until accepted.
// The engine itself is unchanged, including its single-cycle param-memory model.
module neuroedge_mmio_wrapper (
  input  logic clk,
  input  logic rst_n,
  input  logic req_valid,
  input  logic req_write,
  input  logic [7:0] req_addr,
  input  logic [31:0] req_wdata,
  output logic req_ready,
  output logic [31:0] req_rdata
);
  localparam logic [7:0]
    REG_ID       = 8'h00,
    REG_STATUS   = 8'h04,
    REG_WPTR     = 8'h08,
    REG_WDATA    = 8'h0c,
    REG_MPTR     = 8'h10,
    REG_MBIAS    = 8'h14,
    REG_MSHIFT   = 8'h18,
    REG_COMMIT   = 8'h1c,
    REG_INPUT    = 8'h20,
    REG_RESULT   = 8'h24,
    REG_FRAME    = 8'h28,
    REG_COMPLETE = 8'h2c;
  localparam integer WEIGHT_COUNT=44224;
  localparam integer META_COUNT=361;

  logic [15:0] weight_ptr;
  logic [8:0] meta_ptr;
  logic signed [31:0] pending_bias;
  logic committed, fault, bias_valid;
  logic [31:0] completed_frames;

  logic core_in_ready,core_logit_valid,core_frame_done,core_busy;
  logic [5:0] core_logit_index;
  logic signed [15:0] core_logit_data;
  logic [31:0] core_logit_frame;
  logic accept, do_wdata, do_mshift, do_input, do_result;
  logic cfg_permitted;

  assign cfg_permitted = !committed && !core_busy;
  always_comb begin
    req_ready = req_valid;
    if (req_valid && req_write && req_addr == REG_INPUT)
      req_ready = committed && core_in_ready;
    if (req_valid && !req_write && req_addr == REG_RESULT)
      req_ready = core_logit_valid;
    accept = req_valid && req_ready;
    do_wdata = accept && req_write && req_addr==REG_WDATA &&
               cfg_permitted && weight_ptr<WEIGHT_COUNT;
    do_mshift = accept && req_write && req_addr==REG_MSHIFT &&
                cfg_permitted && meta_ptr<META_COUNT && bias_valid;
    do_input = accept && req_write && req_addr==REG_INPUT;
    do_result = accept && !req_write && req_addr==REG_RESULT;
    req_rdata=32'd0;
    case (req_addr)
      REG_ID:       req_rdata=32'h4e453139; // 'NE19'
      REG_STATUS:   req_rdata={28'd0,fault,core_logit_valid,core_busy,committed};
      REG_WPTR:     req_rdata={16'd0,weight_ptr};
      REG_MPTR:     req_rdata={23'd0,meta_ptr};
      REG_FRAME:    req_rdata=core_logit_frame;
      REG_COMPLETE: req_rdata=completed_frames;
      REG_RESULT:   req_rdata={10'd0,core_logit_index,core_logit_data};
      default:      req_rdata=32'd0;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      weight_ptr<=16'd0;
      meta_ptr<=9'd0;
      pending_bias<=32'sd0;
      committed<=1'b0;
      bias_valid<=1'b0;
      fault<=1'b0;
      completed_frames<=32'd0;
    end else begin
      if (core_frame_done) completed_frames<=completed_frames+32'd1;
      if(accept && req_write) begin
        case(req_addr)
          // Address pointers are intentionally READ-ONLY. Sequential loading
          // prevents a fake commit achieved by skipping unwritten addresses.
          REG_WPTR: fault<=1'b1;
          REG_WDATA: if(do_wdata) weight_ptr<=weight_ptr+1'b1;
                     else fault<=1'b1;
          REG_MPTR: fault<=1'b1;
          REG_MBIAS: if(cfg_permitted && meta_ptr<META_COUNT) begin
                       pending_bias<=$signed(req_wdata);
                       bias_valid<=1'b1;
                     end else fault<=1'b1;
          REG_MSHIFT: if(do_mshift) begin
                        meta_ptr<=meta_ptr+1'b1;
                        bias_valid<=1'b0;
                      end else fault<=1'b1;
          REG_COMMIT: if(!committed && weight_ptr==WEIGHT_COUNT &&
                         meta_ptr==META_COUNT && !core_busy && req_wdata[0])
                        committed<=1'b1;
                      else fault<=1'b1;
          default: ;
        endcase
      end
    end
  end

  neuroedge_full_decoder u_core (
    .clk(clk),.rst_n(rst_n),
    .cfg_weight_we(do_wdata),
    .cfg_weight_addr(weight_ptr),
    .cfg_weight_data(req_wdata[7:0]),
    .cfg_meta_we(do_mshift),
    .cfg_meta_addr(meta_ptr),
    .cfg_meta_bias(pending_bias),
    .cfg_meta_shift(req_wdata[4:0]),
    .in_valid(do_input),
    .in_ready(core_in_ready),
    .in_data(req_wdata[7:0]),
    .logit_valid(core_logit_valid),
    .logit_ready(do_result),
    .logit_index(core_logit_index),
    .logit_data(core_logit_data),
    .logit_frame(core_logit_frame),
    .frame_done(core_frame_done),
    .busy(core_busy)
  );
endmodule
