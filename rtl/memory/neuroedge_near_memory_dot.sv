`timescale 1ns/1ps
// Exact sequential INT8 x INT8 dot-product tile with synchronous 1RW weight RAM.
// No in-array multiply: this is NEAR-memory digital compute, not SRAM-CIM.
// Fixed LENGTH input beats per row. Ready/valid stalls are legal at every beat.
// Weight configuration is legal only when index=0 and no request is accepted.
// RAM read initiated on handshake; product consumed on following edge.
module neuroedge_near_memory_dot #(
  parameter integer LENGTH = 512,
  parameter integer ADDR_W = 9
) (
  input wire clk, input wire rst_n,
  input wire cfg_valid,
  output wire cfg_ready,
  input wire [ADDR_W-1:0] cfg_addr,
  input wire signed [7:0] cfg_weight,
  input wire in_valid,
  output wire in_ready,
  input wire signed [7:0] in_data,
  output reg out_valid,
  input wire out_ready,
  output reg signed [31:0] out_sum,
  output wire busy,
  output reg [31:0] accepted_inputs,
  output reg [31:0] accepted_writes
);
  localparam [1:0] S_INPUT=2'd0, S_MAC=2'd1, S_RESULT=2'd2;
  reg [1:0] state;
  reg [ADDR_W-1:0] index;
  reg signed [7:0] x_reg;
  reg signed [31:0] acc;
  wire signed [7:0] w_read;
  wire config_write = cfg_valid && cfg_ready;
  wire take_input = in_valid && in_ready;
  wire [ADDR_W-1:0] ram_addr = config_write ? cfg_addr : index;
  wire signed [15:0] product = $signed(x_reg) * $signed(w_read);
  wire signed [31:0] extended_product = {{16{product[15]}},product};
  assign cfg_ready = (state == S_INPUT) && (index == 0) && !in_valid && (cfg_addr < LENGTH);
  assign in_ready = (state == S_INPUT) && !cfg_valid;
  assign busy = (state != S_INPUT) || (index != 0) || out_valid;
  neuroedge_sram_1rw #(.WIDTH(8), .DEPTH(LENGTH), .ADDR_W(ADDR_W)) weights (
    .clk(clk), .en(config_write || take_input), .we(config_write),
    .addr(ram_addr), .wdata(cfg_weight), .rdata(w_read)
  );
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_INPUT; index <= 0; x_reg <= 0; acc <= 0;
      out_sum <= 0; out_valid <= 0;
      accepted_inputs <= 0; accepted_writes <= 0;
    end else begin
      if (config_write) accepted_writes <= accepted_writes + 1;
      case (state)
        S_INPUT: if (take_input) begin
          x_reg <= in_data;
          accepted_inputs <= accepted_inputs + 1;
          state <= S_MAC;
        end
        S_MAC: begin
          if (index == LENGTH-1) begin
            out_sum <= acc + extended_product;
            out_valid <= 1;
            state <= S_RESULT;
          end else begin
            acc <= acc + extended_product;
            index <= index + 1'b1;
            state <= S_INPUT;
          end
        end
        S_RESULT: if (out_valid && out_ready) begin
          out_valid <= 0;
          acc <= 0;
          index <= 0;
          state <= S_INPUT;
        end
        default: state <= S_INPUT;
      endcase
    end
  end
endmodule
