`timescale 1ns/1ps
module tb_projection;
  reg clk=0;
  always #5 clk=~clk;
  reg rst_n=0;
  reg weight_we=0, feature_we=0, start=0, out_ready=1;
  reg [14:0] weight_addr=0;
  reg signed [7:0] weight_data=0;
  reg [8:0] feature_addr=0;
  reg signed [7:0] feature_data=0;
  wire busy,out_valid,done;
  wire [5:0] out_index;
  wire signed [31:0] out_data;
  reg [7:0] weight_file [0:32767];
  reg [7:0] features_file [0:511];
  reg [31:0] expect_file [0:63];
  string weight_path,feature_path,expect_path;
  integer idx;
  integer seen=0;
  neuroedge_projection dut (.*);

  always @(posedge clk) if (rst_n && out_valid && out_ready) begin
    if (out_index !== seen[5:0]) $fatal(1,"WRONG_ROW expected=%0d got=%0d",seen,out_index);
    if (out_data !== $signed(expect_file[seen]))
      $fatal(1,"PROJECTION_MISMATCH row=%0d expected=%0d actual=%0d",
             seen, $signed(expect_file[seen]),out_data);
    seen = seen + 1;
  end

  initial begin
    if (!$value$plusargs("WEIGHTS=%s",weight_path)) $fatal(1,"MISSING_WEIGHTS");
    if (!$value$plusargs("FEATURES=%s",feature_path)) $fatal(1,"MISSING_FEATURES");
    if (!$value$plusargs("EXPECTED=%s",expect_path)) $fatal(1,"MISSING_EXPECTED");
    $readmemh(weight_path,weight_file);
    $readmemh(feature_path,features_file);
    $readmemh(expect_path,expect_file);
    repeat(5) @(negedge clk);
    rst_n = 1;
    for (idx=0;idx<32768;idx=idx+1) begin
      @(negedge clk);
      weight_we=1;
      weight_addr=idx[14:0];
      weight_data=$signed(weight_file[idx]);
    end
    @(negedge clk); weight_we=0;
    for (idx=0;idx<512;idx=idx+1) begin
      @(negedge clk);
      feature_we=1;
      feature_addr=idx[8:0];
      feature_data=$signed(features_file[idx]);
    end
    @(negedge clk); feature_we=0;
    @(negedge clk); start=1;
    @(negedge clk); start=0;
    wait(done);
    @(negedge clk);
    if (seen != 64) $fatal(1,"INSUFFICIENT_RESULTS expected=64 got=%0d",seen);
    $display("REAL_SPEECH_PROJECTION_RTL_PASS outputs=%0d macs=%0d",seen,32768);
    $finish;
  end
  initial begin
    #2000000;
    $fatal(1,"PROJECTION_TIMEOUT");
  end
endmodule
