`timescale 1ns/1ps
// Self-checking current integration. The existing current real neural input files
// and exact 328-output file must be supplied from the user's WSL tree.
module tb_integrated_offset;
  parameter integer NF=32;
  parameter integer CASE_ID=0; // 0 zero, 1 successful commit, 2 abort/unsafe-deny
  localparam integer NWEIGHT=44224, NMETA=361, INPUTS=512, CLASSES=41;
  localparam integer BUDGET=500000;
  reg clk=0; always #5 clk=~clk;
  reg rst_n=0;
  reg cfg_weight_we=0,cfg_meta_we=0;
  reg [15:0] cfg_weight_addr=0;
  reg signed [7:0] cfg_weight_data=0;
  reg [8:0] cfg_meta_addr=0;
  reg signed [31:0] cfg_meta_bias=0;
  reg [4:0] cfg_meta_shift=0;
  reg feature_valid=0;
  reg signed [7:0] feature_data=0;
  wire feature_ready;
  reg stage_valid=0,abort_stage=0,authorize_update=0;
  reg [8:0] stage_channel=0;
  reg signed [15:0] stage_value=0;
  wire stage_ready;
  wire cfg_ready,request_accepted,request_rejected,pending,write_accepted;
  wire [31:0] accepted_requests,rejected_requests,queued_requests;
  wire [31:0] accepted_writes,stages_accepted,aborted_stages;
  wire bin_in_flight;
  wire [8:0] feature_index;
  wire [31:0] bin_latency_cycles,bins_completed;
  wire logit_valid,frame_done,busy;
  reg logit_ready=0;
  wire [5:0] logit_index;
  wire signed [15:0] logit_data;
  wire [31:0] logit_frame;
  reg [7:0] weight_mem[0:NWEIGHT-1];
  reg [31:0] bias_mem[0:NMETA-1];
  reg [7:0] shift_mem[0:NMETA-1];
  reg [7:0] raw_mem[0:NF*INPUTS-1];
  reg [7:0] cal_mem[0:NF*INPUTS-1];
  reg [15:0] expected_mem[0:NF/4*CLASSES-1];
  reg [15:0] baseline_mem[0:NF/4*CLASSES-1];
  string wp,bp,sp,xp,cp,yp,zp;
  integer commit_channel,commit_value;
  integer i,f,c;
  integer seen=0,changed=0,cores_seen=0,upstream_seen=0;
  integer n_stalls=0,completed=0,all_cycles=0,first_cycle=0;
  integer words_in_bin=0,worst_latency=0,latency;
  integer denied_seen=0;

  neuroedge_speech_pipeline dut (.*);

  // Independent time-dependent backpressure; includes actual S_EMIT stalls.
  always @(negedge clk) begin
    if (!rst_n) logit_ready=0;
    else logit_ready=((($time/10)%13)>=4);
  end

  always @(posedge clk) if(rst_n) begin
    all_cycles=all_cycles+1;
    if(feature_valid && feature_ready) begin
      if(words_in_bin==0) first_cycle=all_cycles;
      words_in_bin=words_in_bin+1;
      upstream_seen=upstream_seen+1;
      if(words_in_bin>INPUTS) $fatal(1,"FEATURE_COUNT_OVERFLOW");
    end
    if(dut.core_in_valid && dut.core_in_ready) begin
      if(cores_seen>=NF*INPUTS) $fatal(1,"EXTRA_CORE_INPUT");
      if(dut.core_in_data !== $signed(cal_mem[cores_seen]))
        $fatal(1,"CALIBRATED_BYTE_MISMATCH idx=%0d exp=%0d got=%0d",
               cores_seen,$signed(cal_mem[cores_seen]),dut.core_in_data);
      cores_seen=cores_seen+1;
    end
    if(logit_valid && !logit_ready) n_stalls=n_stalls+1;
    if(logit_valid && logit_ready) begin
      if(seen >= NF/4*CLASSES) $fatal(1,"TOO_MANY_LOGITS");
      if(logit_frame !== (4*(seen/CLASSES)))
        $fatal(1,"LOGIT_FRAME_MISMATCH idx=%0d frame=%0d",seen,logit_frame);
      if(logit_index !== (seen%CLASSES))
        $fatal(1,"LOGIT_INDEX_MISMATCH idx=%0d got=%0d",seen,logit_index);
      if(logit_data !== $signed(expected_mem[seen]))
        $fatal(1,"LOGIT_MISMATCH idx=%0d expected=%0d got=%0d",
               seen,$signed(expected_mem[seen]),logit_data);
      if(logit_data !== $signed(baseline_mem[seen])) changed=changed+1;
      seen=seen+1;
    end
    if(request_rejected) denied_seen=denied_seen+1;
    if(frame_done) begin
      if(words_in_bin!=INPUTS)
        $fatal(1,"INCOMPLETE_BIN_DONE count=%0d",words_in_bin);
      latency=all_cycles-first_cycle+1;
      if(latency>worst_latency) worst_latency=latency;
      if(latency>=BUDGET) $fatal(1,"LATENCY_OVER_BUDGET %0d",latency);
      completed=completed+1;
      words_in_bin=0;
      #1;
      if(bin_latency_cycles !== latency)
        $fatal(1,"LATENCY_COUNTER_ERROR measured=%0d rtl=%0d",latency,bin_latency_cycles);
      if(bins_completed !== completed)
        $fatal(1,"COMPLETED_COUNTER_ERROR");
    end
  end

  task automatic stage_candidate(input integer ch,input integer value);
  begin
    @(negedge clk);
    if(!stage_ready) $fatal(1,"STAGE_NOT_READY");
    stage_channel=ch[8:0]; stage_value=value;
    stage_valid=1;
    @(negedge clk);stage_valid=0;
  end
  endtask
  task automatic request_once;
  begin
    @(negedge clk);authorize_update=1;
    @(negedge clk);authorize_update=0;
  end
  endtask
  task automatic abort_once;
  begin
    @(negedge clk);abort_stage=1;
    @(negedge clk);abort_stage=0;
  end
  endtask
  task automatic send_one(input reg signed [7:0] v);
  begin
    @(negedge clk);
    feature_data=v;
    feature_valid=1;
    while(feature_ready!==1'b1) @(negedge clk);
    @(posedge clk);
    @(negedge clk);feature_valid=0;
  end
  endtask

  initial begin
    if(!$value$plusargs("WEIGHTS=%s",wp)) $fatal(1,"MISSING_WEIGHTS");
    if(!$value$plusargs("BIASES=%s",bp)) $fatal(1,"MISSING_BIASES");
    if(!$value$plusargs("SHIFTS=%s",sp)) $fatal(1,"MISSING_SHIFTS");
    if(!$value$plusargs("INPUTS=%s",xp)) $fatal(1,"MISSING_INPUTS");
    if(!$value$plusargs("CALIBRATED=%s",cp)) $fatal(1,"MISSING_CALIBRATED");
    if(!$value$plusargs("EXPECTED=%s",yp)) $fatal(1,"MISSING_EXPECTED");
    if(!$value$plusargs("BASELINE=%s",zp)) $fatal(1,"MISSING_BASELINE");
    if(CASE_ID==1) begin
      if(!$value$plusargs("CHANNEL=%d",commit_channel)) $fatal(1,"MISSING_CHANNEL");
      if(!$value$plusargs("OFFSET=%d",commit_value)) $fatal(1,"MISSING_OFFSET");
    end
    $readmemh(wp,weight_mem);
    $readmemh(bp,bias_mem);
    $readmemh(sp,shift_mem);
    $readmemh(xp,raw_mem);
    $readmemh(cp,cal_mem);
    $readmemh(yp,expected_mem);
    $readmemh(zp,baseline_mem);
    repeat(4) @(negedge clk);rst_n=1;
    for(i=0;i<NWEIGHT;i=i+1) begin
      @(negedge clk);
      if(!cfg_ready) $fatal(1,"CONFIG_NOT_READY_WEIGHT idx=%0d",i);
      cfg_weight_addr=i[15:0];cfg_weight_data=$signed(weight_mem[i]);cfg_weight_we=1;
    end
    @(negedge clk);cfg_weight_we=0;
    for(i=0;i<NMETA;i=i+1) begin
      @(negedge clk);
      if(!cfg_ready) $fatal(1,"CONFIG_NOT_READY_META idx=%0d",i);
      cfg_meta_addr=i[8:0];cfg_meta_bias=$signed(bias_mem[i]);
      cfg_meta_shift=shift_mem[i][4:0];cfg_meta_we=1;
    end
    @(negedge clk);cfg_meta_we=0;
    repeat(3) @(negedge clk);

    if(CASE_ID==1) begin
      stage_candidate(commit_channel,commit_value);
      if(!pending || stages_accepted!==1) $fatal(1,"STAGING_FAILURE");
      request_once();
      if(accepted_writes!==1 || accepted_requests!==1 || pending)
        $fatal(1,"COMMIT_FAILED");
    end else if(CASE_ID==2) begin
      stage_candidate(7,-32);
      abort_once();
      if(pending || aborted_stages!==1 || accepted_writes!==0)
        $fatal(1,"ABORT_FAILURE");
      stage_candidate(7,120);
    end
    for(f=0;f<NF;f=f+1) begin
      for(c=0;c<INPUTS;c=c+1) begin
        send_one($signed(raw_mem[f*INPUTS+c]));
        if(CASE_ID==2 && f==0 && (c==0 || c==255 || c==510)) begin
          if(!bin_in_flight) $fatal(1,"PARTIAL_BIN_NOT_TRACKED c=%0d",c);
          if(dut.core_busy) begin
            // For c=0 and c=255 core is still in S_INPUT, i.e.
            // this asserts the current busy-only safety bug is covered.
            $fatal(1,"EXPECTED_CORE_INPUT_IDLE c=%0d",c);
          end
          request_once();
          if(accepted_writes!==0 || !pending)
            $fatal(1,"UNSAFE_WRITE_OCCURRED idx=%0d",c);
        end
      end
      wait(frame_done===1'b1);
      @(negedge clk);
      if(CASE_ID==2 && f==0) begin
        // An authorization cannot turn into a later secret retry.
        abort_once();
        if(pending || accepted_writes!==0 || aborted_stages!==2)
          $fatal(1,"MID_BIN_ABORT_FAILED");
      end
    end
    repeat(6) @(negedge clk);
    if(CASE_ID==2) begin
      request_once();
      repeat(2) @(negedge clk); // registered pulse observed by scoreboard
      if(rejected_requests!==4 || denied_seen!==4 || accepted_writes!==0)
        $fatal(1,"REQUEST_ACCOUNTING_FAIL rejected=%0d observed=%0d",rejected_requests,denied_seen);
    end
    if(seen != NF/4*CLASSES || upstream_seen != NF*INPUTS ||
       cores_seen != NF*INPUTS || completed != NF)
      $fatal(1,"COUNTS_FAIL seen=%0d upstream=%0d core=%0d done=%0d",seen,upstream_seen,cores_seen,completed);
    if(n_stalls==0) $fatal(1,"NO_LOGIT_BACKPRESSURE");
    if(queued_requests!==0) $fatal(1,"UNEXPECTED_QUEUE");
    if(CASE_ID==0) begin
      if(changed!=0 || accepted_writes!=0) $fatal(1,"ZERO_NOT_IDENTICAL");
      $display("ZERO_OFFSET_328_LOGITS_PASS logits=%0d stalls=%0d",seen,n_stalls);
    end else if(CASE_ID==1) begin
      if(changed==0 || accepted_writes!=1 || accepted_requests!=1)
        $fatal(1,"COMMIT_CHANGE_FAIL changed=%0d",changed);
      $display("COMMIT_CHANGES_LOGITS_PASS accepted_writes=%0d changed_logits=%0d",accepted_writes,changed);
    end else if(CASE_ID==2) begin
      if(changed!=0 || accepted_writes!=0 || aborted_stages!=2)
        $fatal(1,"ABORT_NOT_PRESERVED");
      $display("ABORT_DENY_PRESERVE_LOGITS_PASS aborted=%0d denied=%0d",aborted_stages,rejected_requests);
      $display("NO_MID_BIN_WRITE_PASS denied=3");
    end
    $display("BRIDGE_STREAM_RTL_PASS bins=%0d features=%0d stalls=%0d",NF,cores_seen,n_stalls);
    $display("LATENCY_ACCOUNTING_PASS worst_bin_cycles=%0d budget=%0d assumed_clk_hz=25000000",worst_latency,BUDGET);
    $finish;
  end
  initial begin
    #(NF*3500000);
    $fatal(1,"INTEGRATED_TIMEOUT case=%0d seen=%0d bins=%0d",CASE_ID,seen,completed);
  end
endmodule
