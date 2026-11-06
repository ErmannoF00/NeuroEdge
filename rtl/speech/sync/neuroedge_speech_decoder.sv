`timescale 1ns/1ps
// NeuroEdge existing: synchronous single-port SRAM-compatible inference variant.
// Functionally equivalent to existing; every weight/metadata fetch costs a cycle.
// This is memory-interface RTL, NOT a foundry SRAM macro or timing signoff.
// 512 inputs at 20ms/bin -> 64 projection -> DW1 stride2 -> PW1
// -> DW2 stride2 -> PW2 -> 41 CTC logits at bins 0,4,8,...
// FULL INFERENCE, not a synthesized SRAM macro, AFE, CTC beam decoder or learner.
// Memory holds 44,224 INT8 coefficients + 361 INT32 biases + 361 5-bit shifts.
// PROJECT/PW hidden activations [0,127]; DW outputs signed [-128,127].
// Per-row equation: sat_relu((sum(int8*int8) >>> shift) + bias_int32).
// One cycle in S_FETCH obtains weight and metadata, next S_MAC consumes them.
// Never update weights in an in-flight input frame.
// Causal DW history oldest->newest, zero-initialized on reset.
// Input sequence gap can stall freely: no implicit time advancement without 512 data.
module neuroedge_speech_decoder (
  input  logic clk,
  input  logic rst_n,
  input  logic cfg_weight_we,
  input  logic [15:0] cfg_weight_addr,
  input  logic signed [7:0] cfg_weight_data,
  input  logic cfg_meta_we,
  input  logic [8:0] cfg_meta_addr,
  input  logic signed [31:0] cfg_meta_bias,
  input  logic [4:0] cfg_meta_shift,
  input  logic in_valid,
  output logic in_ready,
  input  logic signed [7:0] in_data,
  output logic logit_valid,
  input  logic logit_ready,
  output logic [5:0] logit_index,
  output logic signed [15:0] logit_data,
  output logic [31:0] logit_frame,
  output logic frame_done,
  output logic busy
);
  localparam integer HIDDEN=64;
  localparam integer INPUTS=512;
  localparam integer CLASSES=41;
  localparam integer WEIGHT_COUNT=44224;
  localparam integer META_COUNT=361;
  localparam integer PROJECT=0, DW1=1, PW1=2, DW2=3, PW2=4, HEAD=5;

  // The parameter store is a single-port, synchronous-read abstraction.
  // A read requested in S_FETCH is consumed one cycle later in S_MAC.
  logic [7:0] weight_read;
  logic [36:0] meta_read; // {signed 32-bit bias, 5-bit unsigned shift}
  logic signed [7:0] inputs[0:INPUTS-1];
  logic signed [7:0] projected[0:HIDDEN-1];
  logic signed [7:0] depth1[0:HIDDEN-1];
  logic signed [7:0] point1[0:HIDDEN-1];
  logic signed [7:0] depth2[0:HIDDEN-1];
  logic signed [7:0] point2[0:HIDDEN-1];
  // Time 0 is oldest, time 4 newest. Reset supplies left-zero padding.
  logic signed [7:0] history0[0:4][0:HIDDEN-1];
  logic signed [7:0] history1[0:4][0:HIDDEN-1];

  typedef enum logic [2:0] {S_INPUT,S_FETCH,S_MAC,S_SHIFT0,S_SHIFT1,S_EMIT} state_t;
  state_t state;
  logic [2:0] phase;
  logic [8:0] input_index;
  logic [8:0] tap;
  logic [5:0] row;
  logic signed [31:0] accumulator;
  logic [31:0] frame_counter;

  logic [15:0] weight_base;
  logic [8:0] meta_base;
  logic [9:0] fan_in;
  logic [6:0] fan_out;
  integer weight_index;
  integer meta_index;
  logic signed [7:0] operand;
  logic signed [15:0] product;
  logic signed [31:0] product32;
  logic signed [31:0] next_sum;
  logic signed [31:0] shifted_sum;
  logic signed [63:0] biased_sum;

  // Weight and metadata stores are fully configured in S_INPUT only.
  // Reads are mutually exclusive with writes; this matches a 1RW port.
  wire weight_write = (state == S_INPUT) && cfg_weight_we &&
                      (cfg_weight_addr < WEIGHT_COUNT);
  wire meta_write = (state == S_INPUT) && cfg_meta_we &&
                    (cfg_meta_addr < META_COUNT);
  wire weight_fetch = (state == S_FETCH);
  wire meta_fetch = (state == S_FETCH);
  wire [15:0] sram_weight_addr = weight_write ? cfg_weight_addr : weight_index[15:0];
  wire [8:0] sram_meta_addr = meta_write ? cfg_meta_addr : meta_index[8:0];

  neuroedge_sram_1rw #(.WIDTH(8), .DEPTH(WEIGHT_COUNT), .ADDR_W(16))
    u_weight_mem (
      .clk(clk), .en(weight_write | weight_fetch), .we(weight_write),
      .addr(sram_weight_addr), .wdata(cfg_weight_data), .rdata(weight_read)
    );
  neuroedge_sram_1rw #(.WIDTH(37), .DEPTH(META_COUNT), .ADDR_W(9))
    u_meta_mem (
      .clk(clk), .en(meta_write | meta_fetch), .we(meta_write),
      .addr(sram_meta_addr), .wdata({cfg_meta_bias,cfg_meta_shift}),
      .rdata(meta_read)
    );

  function automatic signed [7:0] clamp_relu8(input signed [63:0] v);
    if (v <= 0) clamp_relu8=8'sd0;
    else if (v >= 127) clamp_relu8=8'sd127;
    else clamp_relu8=v[7:0];
  endfunction
  function automatic signed [7:0] clamp_signed8(input signed [63:0] v);
    if (v < -128) clamp_signed8=8'sh80;
    else if (v > 127) clamp_signed8=8'sd127;
    else clamp_signed8=v[7:0];
  endfunction
  function automatic signed [15:0] clamp_int16(input signed [63:0] v);
    if (v < -32768) clamp_int16=16'sh8000;
    else if (v > 32767) clamp_int16=16'sd32767;
    else clamp_int16=v[15:0];
  endfunction

  always_comb begin
    case(phase)
      PROJECT: begin weight_base=16'd0;     meta_base=9'd0;   fan_in=10'd512;fan_out=7'd64;end
      DW1:     begin weight_base=16'd32768; meta_base=9'd64;  fan_in=10'd5;  fan_out=7'd64;end
      PW1:     begin weight_base=16'd33088; meta_base=9'd128; fan_in=10'd64; fan_out=7'd64;end
      DW2:     begin weight_base=16'd37184; meta_base=9'd192; fan_in=10'd5;  fan_out=7'd64;end
      PW2:     begin weight_base=16'd37504; meta_base=9'd256; fan_in=10'd64; fan_out=7'd64;end
      default: begin weight_base=16'd41600; meta_base=9'd320; fan_in=10'd64; fan_out=7'd41;end
    endcase
    weight_index={16'd0,weight_base}+({26'd0,row}*{22'd0,fan_in})+{23'd0,tap};
    meta_index={23'd0,meta_base}+{26'd0,row};
    operand=8'sd0;
    case(phase)
      PROJECT: operand=inputs[tap];
      DW1:     operand=history0[tap][row];
      PW1:     operand=depth1[tap];
      DW2:     operand=history1[tap][row];
      PW2:     operand=depth2[tap];
      default: operand=point2[tap];
    endcase
    product=$signed(weight_read)*$signed(operand);
    product32={{16{product[15]}},product};
    next_sum=accumulator+product32;
    shifted_sum=next_sum >>> meta_read[4:0];
    biased_sum=$signed(shifted_sum)+$signed(meta_read[36:5]);
  end

  always_comb begin
    in_ready=(state==S_INPUT) && !cfg_weight_we && !cfg_meta_we;
    busy=(state!=S_INPUT);
  end

  integer i,j;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      state<=S_INPUT;
      phase<=PROJECT;
      input_index<=0;
      tap<=0;
      row<=0;
      accumulator<=0;
      frame_counter<=0;
      logit_valid<=0;
      logit_index<=0;
      logit_data<=0;
      logit_frame<=0;
      frame_done<=0;
      for(i=0;i<5;i=i+1) begin
        for(j=0;j<HIDDEN;j=j+1) begin
          history0[i][j]<=0;
          history1[i][j]<=0;
        end
      end
    end else begin
      frame_done<=0;
      // Configuration writes and reads live in explicit 1RW memory blocks.
      case(state)
        S_INPUT: if(in_valid && in_ready) begin
          inputs[input_index]<=in_data;
          if(input_index==INPUTS-1) begin
            input_index<=0;
            row<=0;
            tap<=0;
            accumulator<=0;
            phase<=PROJECT;
            state<=S_FETCH;
          end else input_index<=input_index+1'b1;
        end
        S_FETCH: state<=S_MAC;
        S_MAC: begin
          if(tap==fan_in-1'b1) begin
            // Save final result for this row using completed accumulation.
            case(phase)
              PROJECT: projected[row]<=clamp_relu8(biased_sum);
              DW1: depth1[row]<=clamp_signed8(biased_sum);
              PW1: point1[row]<=clamp_relu8(biased_sum);
              DW2: depth2[row]<=clamp_signed8(biased_sum);
              PW2: point2[row]<=clamp_relu8(biased_sum);
              default: begin
                logit_data<=clamp_int16(biased_sum);
                logit_index<=row;
                logit_frame<=frame_counter;
                logit_valid<=1'b1;
              end
            endcase
            if(phase==HEAD) begin
              state<=S_EMIT;
            end else if(row==fan_out-1'b1) begin
              row<=0;
              tap<=0;
              accumulator<=0;
              case(phase)
                PROJECT: state<=S_SHIFT0;
                DW1: begin phase<=PW1;state<=S_FETCH;end
                PW1: state<=S_SHIFT1;
                DW2: begin phase<=PW2;state<=S_FETCH;end
                PW2: begin phase<=HEAD;state<=S_FETCH;end
                default: state<=S_INPUT;
              endcase
            end else begin
              row<=row+1'b1;
              tap<=0;
              accumulator<=0;
              state<=S_FETCH;
            end
          end else begin
            accumulator<=next_sum;
            tap<=tap+1'b1;
            state<=S_FETCH;
          end
        end
        S_SHIFT0: begin
          history0[0][row]<=history0[1][row];
          history0[1][row]<=history0[2][row];
          history0[2][row]<=history0[3][row];
          history0[3][row]<=history0[4][row];
          history0[4][row]<=projected[row];
          if(row==HIDDEN-1) begin
            row<=0;tap<=0;accumulator<=0;
            if(!frame_counter[0]) begin
              phase<=DW1;
              state<=S_FETCH;
            end else begin
              frame_counter<=frame_counter+1'b1;
              frame_done<=1'b1;
              state<=S_INPUT;
            end
          end else row<=row+1'b1;
        end
        S_SHIFT1: begin
          history1[0][row]<=history1[1][row];
          history1[1][row]<=history1[2][row];
          history1[2][row]<=history1[3][row];
          history1[3][row]<=history1[4][row];
          history1[4][row]<=point1[row];
          if(row==HIDDEN-1) begin
            row<=0;tap<=0;accumulator<=0;
            if(frame_counter[1:0]==0) begin
              phase<=DW2;
              state<=S_FETCH;
            end else begin
              frame_counter<=frame_counter+1'b1;
              frame_done<=1'b1;
              state<=S_INPUT;
            end
          end else row<=row+1'b1;
        end
        S_EMIT: if(logit_valid && logit_ready) begin
          logit_valid<=1'b0;
          tap<=0;
          accumulator<=0;
          if(row==CLASSES-1) begin
            row<=0;
            frame_counter<=frame_counter+1'b1;
            frame_done<=1'b1;
            state<=S_INPUT;
          end else begin
            row<=row+1'b1;
            state<=S_FETCH;
          end
        end
        default: state<=S_INPUT;
      endcase
    end
  end
endmodule
