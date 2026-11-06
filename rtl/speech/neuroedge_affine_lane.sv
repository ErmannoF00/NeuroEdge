`timescale 1ns/1ps
// Hardware datapath only. Coefficients supplied by future SRAM/CSR subsystem.
// x is signed INT8 (scaled normalized neural feature); coefficient format:
// gain_q = round(gain * 2^12), offset_q = round(offset_in_INT8_units * 2^12).
// y = saturate_int8((x*gain_q + offset_q) >>> 12), arithmetic floor rounding.
// 1-stage valid/ready pipeline; no on-chip optimization/backprop here.
module neuroedge_affine_lane #(
    parameter integer FRACTION_BITS = 12
)(
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,
    output wire in_ready,
    input  wire signed [7:0] in_sample,
    input  wire signed [23:0] in_gain_q,
    input  wire signed [23:0] in_offset_q,
    output reg  out_valid,
    input  wire out_ready,
    output reg signed [7:0] out_sample
);
    wire signed [47:0] product = $signed(in_sample) * $signed(in_gain_q);
    wire signed [47:0] sum = product + $signed(in_offset_q);
    wire signed [47:0] shifted = sum >>> FRACTION_BITS;
    wire signed [7:0] saturated =
        (shifted > 48'sd127) ? 8'sd127 :
        (shifted < -48'sd128) ? 8'h80 : shifted[7:0];

    assign in_ready = (~out_valid) | out_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_sample <= 8'sd0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid)
                out_sample <= saturated;
        end
    end
endmodule
