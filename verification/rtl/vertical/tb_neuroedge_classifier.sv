`timescale 1ns / 1ps
`include "model_params.vh"
module tb_neuroedge_classifier;
    localparam integer N = `MODEL_N;
    localparam integer NV = `MODEL_NUM_VECTORS;
    reg clk=0;
    always #5 clk = ~clk;
    reg rst_n=0, in_valid=0, out_ready=0;
    reg signed [7:0] feature=0;
    wire in_ready, out_valid, out_class;
    wire signed [31:0] out_score;
    reg [N*8+31:0] vectors [0:NV-1];
    reg signed [31:0] expected;
    integer row, col;
    integer checked=0;

    neuroedge_linear_classifier #(
        .N_FEATURES(N),
        .BIAS(`MODEL_BIAS),
        .WEIGHT_FILE("weights.mem")
    ) dut (
        .clk_i(clk), .rst_ni(rst_n),
        .in_valid_i(in_valid), .in_ready_o(in_ready),
        .in_feature_i(feature),
        .out_valid_o(out_valid), .out_ready_i(out_ready),
        .out_score_o(out_score), .out_class_o(out_class)
    );
    initial begin
        $readmemh("vectors.mem", vectors);
        repeat(3) @(negedge clk);
        rst_n=1;
        for(row=0; row<NV; row=row+1) begin
            for(col=0; col<N; col=col+1) begin
                @(negedge clk);
                if (!in_ready) $fatal(1, "INPUT_NOT_READY at row %0d col %0d",row,col);
                in_valid=1;
                feature = $signed(vectors[row][32+(N-col)*8-1 -:8]);
                @(posedge clk);
                #1;
            end
            @(negedge clk);
            in_valid=0;
            expected = $signed(vectors[row][31:0]);
            if (!out_valid) $fatal(1,"MISSING_RESULT row=%0d",row);
            if (out_score !== expected)
                $fatal(1,"SCORE_MISMATCH row=%0d got=%0d expect=%0d",row,out_score,expected);
            if (out_class !== (expected >= 0))
                $fatal(1,"CLASS_MISMATCH row=%0d",row);
            checked=checked+1;
            out_ready=1;
            @(posedge clk);
            #1;
            @(negedge clk);
            out_ready=0;
        end
        $display("RTL_EQUIVALENCE_PASS vectors=%0d",checked);
        $finish;
    end
    initial begin
        #1000000;
        $fatal(1, "TIMEOUT");
    end
endmodule
