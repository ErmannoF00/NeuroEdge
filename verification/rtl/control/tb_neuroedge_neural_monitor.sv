`timescale 1ns/1ps
`ifndef NE_FRAMES
`define NE_FRAMES 16
`endif
`ifndef NE_HIGH
`define NE_HIGH 64
`endif
`ifndef NE_LOW
`define NE_LOW 40
`endif
`ifndef NE_MAG
`define NE_MAG 32
`endif
module tb_neuroedge_neural_monitor;
  localparam integer INPUTS=512;
  localparam integer FRAMES=`NE_FRAMES;
  reg clk=0;
  always #5 clk=~clk;
  reg rst_n=0;
  reg feature_valid=0;
  reg signed [7:0] feature_data=0;
  wire score_valid;
  wire [15:0] score;
  wire active, request;
  wire [15:0] writes;
  wire [31:0] observed, suppressed;
  reg [7:0] feature_memory[0:FRAMES*INPUTS-1];
  reg [1023:0] source_file, expected_file;
  integer fd, read_count, bin_id, feature_id;
  integer exp_score, exp_active, exp_request, exp_writes, exp_suppressed;
  integer requests=0;

  neuroedge_neural_activity_monitor #(
    .MAG_THRESHOLD(`NE_MAG),.HIGH_THRESHOLD(`NE_HIGH),.LOW_THRESHOLD(`NE_LOW)
  ) dut(
    .clk(clk),.rst_n(rst_n),.feature_valid(feature_valid),
    .feature_data(feature_data),.score_valid(score_valid),.score(score),
    .drift_active(active),.update_request(request),
    .requests_in_window(writes),.observed_bins(observed),
    .suppressed_requests(suppressed)
  );
  initial begin
    if(!$value$plusargs("FEATURES=%s",source_file)) $fatal(1,"MISSING_FEATURES");
    if(!$value$plusargs("EXPECTED=%s",expected_file)) $fatal(1,"MISSING_EXPECTED");
    $readmemh(source_file,feature_memory);
    fd=$fopen(expected_file,"r");
    if(fd==0) $fatal(1,"CANNOT_OPEN_EXPECTED");
    repeat (4) @(negedge clk);
    rst_n=1;
    for(bin_id=0;bin_id<FRAMES;bin_id=bin_id+1) begin
      for(feature_id=0;feature_id<INPUTS;feature_id=feature_id+1) begin
        @(negedge clk);
        feature_valid=1;
        feature_data=$signed(feature_memory[bin_id*INPUTS+feature_id]);
      end
      @(negedge clk);
      feature_valid=0;
      if(!score_valid) $fatal(1,"SCORE_NOT_VALID bin=%0d",bin_id);
      read_count=$fscanf(fd,"%d %d %d %d %d\n",
                         exp_score,exp_active,exp_request,exp_writes,exp_suppressed);
      if(read_count!=5) $fatal(1,"EXPECTED_VECTOR_SHORT bin=%0d",bin_id);
      if(score!==exp_score)
        $fatal(1,"SCORE_MISMATCH bin=%0d expected=%0d got=%0d",bin_id,exp_score,score);
      // Score-valid was asserted by last feature; controller samples it now.
      @(posedge clk); #1;
      if(active !== (exp_active != 0) || request !== (exp_request != 0) ||
         writes !== exp_writes || suppressed !== exp_suppressed || observed !== bin_id+1)
        $fatal(1,"POLICY_MISMATCH bin=%0d expected=%0d,%0d,%0d,%0d got=%0d,%0d,%0d,%0d",
               bin_id,exp_active,exp_request,exp_writes,exp_suppressed,
               active,request,writes,suppressed);
      requests=requests+exp_request;
    end
    $fclose(fd);
    $display("NEURAL_ACTIVITY_RTL_PASS bins=%0d features=%0d requests=%0d",
             FRAMES,FRAMES*INPUTS,requests);
    $finish;
  end
endmodule
