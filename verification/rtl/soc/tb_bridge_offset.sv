`timescale 1ns/1ps
// Memory+bridge with artificial consumer pauses, independent of neural decoder.
module tb_bridge_offset;
  reg clk=0; always #5 clk=~clk;
  reg rst_n=0,feature_valid=0,stage_valid=0,abort_stage=0,update_request=0;
  reg signed [7:0] feature_data=0;
  wire feature_ready;
  reg [8:0] stage_channel=0;
  reg signed [15:0] stage_value=0;
  wire stage_ready,read_enable,core_in_valid,bin_in_flight,bridge_empty;
  wire [8:0] read_channel,feature_index;
  wire signed [15:0] read_data;
  wire signed [7:0] core_in_data;
  wire pending,write_accepted;
  wire [31:0] accepted_writes,rejected_requests,stages_accepted,aborted_stages;
  wire [31:0] bin_latency_cycles,bins_completed;
  reg core_frame_done=0;
  reg [15:0] lfsr=16'h1ace;
  wire core_in_ready = rst_n && (lfsr[0] || lfsr[3]);
  integer tick=0,stream_count=0,byte_count=0,stalled=0;
  reg previous_stall=0;
  reg signed [7:0] previous_stall_data=0;
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) lfsr<=16'h1ace;
    else lfsr<={lfsr[14:0],lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
  end
  integer input_count=0;
  integer c;
  reg signed [7:0] expect_byte;
  neuroedge_offset_bank bank (
    .clk(clk),.rst_n(rst_n),.stage_valid(stage_valid),.stage_ready(stage_ready),
    .stage_channel(stage_channel),.stage_value(stage_value),
    .abort_stage(abort_stage),.update_request(update_request),
    .decoder_idle(!bin_in_flight && bridge_empty && !read_enable && !core_frame_done),
    .read_enable(read_enable),.read_channel(read_channel),.read_data(read_data),
    .pending(pending),.write_accepted(write_accepted),
    .accepted_writes(accepted_writes),.rejected_requests(rejected_requests),
    .stages_accepted(stages_accepted),.aborted_stages(aborted_stages)
  );
  neuroedge_offset_stream_bridge bridge (
    .clk(clk),.rst_n(rst_n),.feature_valid(feature_valid),
    .feature_ready(feature_ready),.feature_data(feature_data),
    .offset_read_enable(read_enable),.offset_read_channel(read_channel),
    .offset_read_data(read_data),.core_in_valid(core_in_valid),
    .core_in_ready(core_in_ready),.core_in_data(core_in_data),
    .core_frame_done(core_frame_done),.allow_features(1'b1),
    .bin_in_flight(bin_in_flight),.bridge_empty(bridge_empty),
    .feature_index(feature_index),.bin_latency_cycles(bin_latency_cycles),
    .bins_completed(bins_completed)
  );
  always @(posedge clk) if (rst_n) begin
    tick=tick+1;
    if(previous_stall && (!core_in_valid || core_in_data !== previous_stall_data))
      $fatal(1,"BRIDGE_VALID_DATA_NOT_HELD_UNDER_BACKPRESSURE");
    previous_stall = core_in_valid && !core_in_ready;
    if(previous_stall) previous_stall_data = core_in_data;
    if(core_in_valid && !core_in_ready) stalled=stalled+1;
    if(feature_valid && feature_ready) input_count=input_count+1;
    if(core_in_valid && core_in_ready) begin
      expect_byte=(byte_count%512)-128;
      if (stream_count==1 && byte_count%512==7) expect_byte=127;
      if (stream_count==1 && byte_count%512==9) expect_byte=-128;
      if (core_in_data !== expect_byte)
        $fatal(1,"BRIDGE_BYTE_FAIL bin=%0d channel=%0d got=%0d expected=%0d",
               stream_count,byte_count%512,core_in_data,expect_byte);
      byte_count=byte_count+1;
      if(byte_count%512==0) begin
        core_frame_done<=1;
        stream_count=stream_count+1;
      end
    end else core_frame_done<=0;
  end
  task automatic send(input integer v);
  begin
    @(negedge clk);feature_valid=1;feature_data=v;
    while(!feature_ready) @(negedge clk);
    @(posedge clk);
    @(negedge clk);feature_valid=0;
  end
  endtask
  task automatic stage(input integer ch,input integer value);
  begin
    @(negedge clk);
    if(!stage_ready) $fatal(1,"BRIDGE_STAGE_NOT_READY");
    stage_channel=ch;stage_value=value;stage_valid=1;
    @(negedge clk);stage_valid=0;
  end
  endtask
  task automatic commit;
  begin
    @(negedge clk);update_request=1;
    @(negedge clk);update_request=0;
  end
  endtask
  initial begin
    repeat(4) @(negedge clk);rst_n=1;
    for(c=0;c<512;c=c+1) send(c-128);
    wait(bins_completed==1);
    repeat(3) @(negedge clk);
    stage(7,32767);commit();
    stage(9,-32768);commit();
    if(accepted_writes!==2) $fatal(1,"BRIDGE_COMMITS_FAILED");
    for(c=0;c<512;c=c+1) send(c-128);
    wait(bins_completed==2);
    repeat(4) @(negedge clk);
    if(stream_count!=2 || byte_count!=1024 || input_count!=1024 || stalled==0)
      $fatal(1,"BRIDGE_STALL_OR_COUNTS_FAILED");
    if(accepted_writes!=2 || rejected_requests!=0)
      $fatal(1,"BRIDGE_WRITE_COUNTS_FAILED");
    $display("BRIDGE_UNIT_RTL_PASS bins=2 features=1024 stalls=%0d commits=2",stalled);
    $finish;
  end
  initial begin #500000; $fatal(1,"BRIDGE_UNIT_TIMEOUT"); end
endmodule
