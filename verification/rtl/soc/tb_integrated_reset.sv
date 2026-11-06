`timescale 1ns/1ps
// Directed asynchronous reset during a partially accepted neural bin.
// No model parameter load needed: checks front-end bytes and reset validity,
// not decoder outputs (those would require the unavailable real vectors).
module tb_integrated_reset;
  reg clk=0; always #5 clk=~clk;
  reg rst_n=0;
  reg cfg_weight_we=0,cfg_meta_we=0;
  reg [15:0] cfg_weight_addr=0;
  reg signed [7:0] cfg_weight_data=0;
  reg [8:0] cfg_meta_addr=0;
  reg signed [31:0] cfg_meta_bias=0;
  reg [4:0] cfg_meta_shift=0;
  wire cfg_ready;
  reg feature_valid=0;
  wire feature_ready;
  reg signed [7:0] feature_data=0;
  reg stage_valid=0;
  wire stage_ready;
  reg [8:0] stage_channel=0;
  reg signed [15:0] stage_value=0;
  reg abort_stage=0,authorize_update=0;
  wire request_accepted,request_rejected;
  wire [31:0] accepted_requests,rejected_requests,queued_requests;
  wire pending,write_accepted;
  wire [31:0] accepted_writes,stages_accepted,aborted_stages;
  wire bin_in_flight;
  wire [8:0] feature_index;
  wire [31:0] bin_latency_cycles,bins_completed;
  wire logit_valid,frame_done,busy;
  reg logit_ready=1;
  wire [5:0] logit_index;
  wire signed [15:0] logit_data;
  wire [31:0] logit_frame;
  integer matched=0;
  neuroedge_speech_pipeline dut (.*);

  task automatic stage(input integer ch,input integer val);
  begin
    @(negedge clk);
    if(!stage_ready) $fatal(1,"RESET_STAGE_NOT_READY");
    stage_channel=ch;stage_value=val;stage_valid=1;
    @(negedge clk);stage_valid=0;
  end
  endtask
  task automatic request_commit;
  begin
    @(negedge clk);authorize_update=1;
    @(negedge clk);authorize_update=0;
  end
  endtask
  task automatic feed(input integer v);
  begin
    @(negedge clk);feature_valid=1;feature_data=v;
    while(!feature_ready) @(negedge clk);
    @(posedge clk);
    @(negedge clk);feature_valid=0;
  end
  endtask
  always @(posedge clk) if(rst_n && dut.core_in_valid && dut.core_in_ready) begin
    if(matched==0 && dut.core_in_data !== 8'sd110)
      $fatal(1,"RESET_PRECOMMIT_VALUE_MISMATCH got=%0d",dut.core_in_data);
    if(matched==1 && dut.core_in_data !== 8'sd10)
      $fatal(1,"RESET_VALID_BIT_NOT_CLEARED got=%0d",dut.core_in_data);
    matched=matched+1;
  end
  initial begin
    repeat(4) @(negedge clk);rst_n=1;
    stage(0,100);
    request_commit();
    if(accepted_writes!==1 || pending)
      $fatal(1,"RESET_SETUP_COMMIT_FAIL");
    feed(10);
    wait(matched==1);
    stage(1,50);
    if(!pending || !bin_in_flight || dut.core_busy)
      $fatal(1,"RESET_EXPECTED_PARTIAL_INPUT");
    // asynchronous reset before 512th input: no phantom completed bin.
    @(negedge clk);rst_n=0;
    #1;
    if(accepted_writes!==0 || pending || bin_in_flight ||
       feature_index!==0 || accepted_requests!==0 || rejected_requests!==0)
      $fatal(1,"RESET_DID_NOT_CLEAR_TRANSACTIONAL_STATE");
    repeat(3) @(negedge clk);rst_n=1;
    feed(10);
    wait(matched==2);
    if(accepted_writes!==0 || bins_completed!==0)
      $fatal(1,"RESET_PHANTOM_COMPLETION_OR_COMMIT");
    $display("RESET_RECOVERY_RTL_PASS pending_clear=1 valid_clear=1 partial_clear=1");
    $finish;
  end
  initial begin #100000; $fatal(1,"RESET_TIMEOUT"); end
endmodule
