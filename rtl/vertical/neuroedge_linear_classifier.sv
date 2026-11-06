// SPDX-License-Identifier: MIT
// NeuroEdge v1 signed-INT8 serial dot-product classifier, signed-INT32 score.
// One accepted feature per clock; output is held until out_ready_i.
// Values at the interface are integer features AFTER software preprocessing.
module neuroedge_linear_classifier #(
    parameter integer N_FEATURES = 8,
    parameter signed [31:0] BIAS = 32'sd0,
    parameter WEIGHT_FILE = "weights.mem"
) (
    input  wire clk_i,
    input  wire rst_ni,
    input  wire in_valid_i,
    output wire in_ready_o,
    input  wire signed [7:0] in_feature_i,
    output reg  out_valid_o,
    input  wire out_ready_i,
    output reg  signed [31:0] out_score_o,
    output reg  out_class_o
);
    localparam integer IDX_BITS = (N_FEATURES < 2) ? 1 : $clog2(N_FEATURES);
    reg [IDX_BITS-1:0] index_q;
    reg signed [31:0] accumulated_q;
    reg signed [7:0] weights [0:N_FEATURES-1];

    // For a production SRAM implementation, replace the inferred ROM with an
    // SRAM macro, perform reset/initialization over firmware, and characterize
    // timing/power using technology-specific macro views.
    initial $readmemh(WEIGHT_FILE, weights);

    wire signed [15:0] product = in_feature_i * weights[index_q];
    wire signed [31:0] extended_product = {{16{product[15]}}, product};
    wire signed [31:0] sum_next = accumulated_q + extended_product;
    wire signed [31:0] final_score = sum_next + BIAS;
    assign in_ready_o = ~out_valid_o | out_ready_i;

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            index_q       <= {IDX_BITS{1'b0}};
            accumulated_q <= 32'sd0;
            out_valid_o   <= 1'b0;
            out_score_o   <= 32'sd0;
            out_class_o   <= 1'b0;
        end else begin
            if (out_valid_o && out_ready_i)
                out_valid_o <= 1'b0;
            if (in_valid_i && in_ready_o) begin
                if (index_q == N_FEATURES-1) begin
                    out_score_o   <= final_score;
                    out_class_o   <= ~final_score[31];  // >=0 -> class 1
                    out_valid_o   <= 1'b1;
                    index_q       <= {IDX_BITS{1'b0}};
                    accumulated_q <= 32'sd0;
                end else begin
                    accumulated_q <= sum_next;
                    index_q       <= index_q + 1'b1;
                end
            end
        end
    end
endmodule
