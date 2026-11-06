`timescale 1ns/1ps
// NeuroEdge first-layer INT8 GEMV, 512 inputs x 64 outputs.
// Single sequential MAC lane; explicitly an SRAM-abstract behavioral memory,
// NOT a characterized SRAM macro. This is a reference building block.
// Signed dot products: y[j] = sum_{i=0}^{511} x_q[i] * w_q[j][i].
// Weights/activations are explicitly written via configuration ports.
// No rounding, bias or ReLU included (they belong to later pipeline stages).
module neuroedge_projection #(
  parameter int INPUTS = 512,
  parameter int OUTPUTS = 64
) (
  input  logic                clk,
  input  logic                rst_n,
  input  logic                weight_we,
  input  logic [14:0]         weight_addr,
  input  logic signed [7:0]   weight_data,
  input  logic                feature_we,
  input  logic [8:0]          feature_addr,
  input  logic signed [7:0]   feature_data,
  input  logic                start,
  output logic                busy,
  output logic                out_valid,
  input  logic                out_ready,
  output logic [5:0]          out_index,
  output logic signed [31:0]  out_data,
  output logic                done
);
  // V0.4 specialization: address widths enforce this exact shape.
  localparam int TOTAL_WEIGHTS = INPUTS * OUTPUTS;
  logic signed [7:0] weights [0:TOTAL_WEIGHTS-1];
  logic signed [7:0] features [0:INPUTS-1];
  typedef enum logic [1:0] {S_IDLE, S_MAC, S_EMIT} state_t;
  state_t state;
  logic [8:0] in_idx;
  logic [5:0] row_idx;
  logic signed [31:0] accumulator;
  wire [14:0] mac_addr = {row_idx, in_idx};
  wire signed [15:0] product = $signed(features[in_idx]) * $signed(weights[mac_addr]);
  wire signed [31:0] product32 = {{16{product[15]}},product};

  assign busy = (state != S_IDLE);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      in_idx <= '0;
      row_idx <= '0;
      accumulator <= '0;
      out_valid <= 1'b0;
      out_index <= '0;
      out_data <= '0;
      done <= 1'b0;
    end else begin
      done <= 1'b0;
      // Disallow writes during compute: no partially programmed inference.
      if (state == S_IDLE) begin
        if (weight_we) weights[weight_addr] <= weight_data;
        if (feature_we) features[feature_addr] <= feature_data;
      end
      case (state)
        S_IDLE: if (start && !weight_we && !feature_we) begin
          accumulator <= '0;
          row_idx <= '0;
          in_idx <= '0;
          out_valid <= 1'b0;
          state <= S_MAC;
        end
        S_MAC: begin
          if (in_idx == INPUTS-1) begin
            out_data <= accumulator + product32;
            out_index <= row_idx;
            out_valid <= 1'b1;
            state <= S_EMIT;
          end else begin
            accumulator <= accumulator + product32;
            in_idx <= in_idx + 1'b1;
          end
        end
        S_EMIT: if (out_valid && out_ready) begin
          out_valid <= 1'b0;
          if (row_idx == OUTPUTS-1) begin
            state <= S_IDLE;
            done <= 1'b1;
          end else begin
            row_idx <= row_idx + 1'b1;
            in_idx <= '0;
            accumulator <= '0;
            state <= S_MAC;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
