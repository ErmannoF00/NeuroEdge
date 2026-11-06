`timescale 1ns/1ps
module tb_full_decoder_sync;
  parameter integer NF=8;
  localparam integer NWEIGHT=44224, NMETA=361, INPUTS=512, CLASSES=41;
  reg clk=0;
  always #5 clk=~clk;
  reg rst_n=0;
  reg cfg_weight_we=0,cfg_meta_we=0,in_valid=0,logit_ready=0;
  reg [15:0] cfg_weight_addr=0;
  reg signed [7:0] cfg_weight_data=0;
  reg [8:0] cfg_meta_addr=0;
  reg signed [31:0] cfg_meta_bias=0;
  reg [4:0] cfg_meta_shift=0;
  reg signed [7:0] in_data=0;
  wire in_ready,logit_valid,frame_done,busy;
  wire [5:0] logit_index;
  wire signed [15:0] logit_data;
  wire [31:0] logit_frame;
  reg [7:0] weights_file[0:NWEIGHT-1];
  reg [31:0] biases_file[0:NMETA-1];
  reg [7:0] shifts_file[0:NMETA-1];
  reg [7:0] input_file[0:NF*INPUTS-1];
  reg [15:0] expected_file[0:((NF+3)/4)*CLASSES-1];
  string wpath,bpath,spath,xpath,ypath;
  integer i,f,c;
  integer seen=0;
  integer frames_completed=0;
  integer all_cycles=0;
  integer input_first_cycle=0;
  integer input_words_this_bin=0;
  integer measured_cycle_latency=0;
  integer worst_cycle_latency=0;
  localparam integer CYCLES_PER_BIN_AT_25MHZ=500000;
  // Detect a new neural bin from accepted input handshakes.
  always @(posedge clk) if(rst_n) begin
    all_cycles=all_cycles+1;
    if(in_valid && in_ready) begin
      if(input_words_this_bin==0)
        input_first_cycle=all_cycles;
      input_words_this_bin=input_words_this_bin+1;
      if(input_words_this_bin>INPUTS)
        $fatal(1,"SYNC_SRAM_INPUT_COUNT_OVERFLOW count=%0d",input_words_this_bin);
    end
    if(frame_done) begin
      if(input_words_this_bin!=INPUTS)
        $fatal(1,"SYNC_SRAM_INPUT_COUNT_FAIL count=%0d",input_words_this_bin);
      measured_cycle_latency=all_cycles-input_first_cycle+1;
      if(measured_cycle_latency>worst_cycle_latency)
        worst_cycle_latency=measured_cycle_latency;
      $display("SYNC_SRAM_BIN_LATENCY frame=%0d cycles=%0d",frames_completed,measured_cycle_latency);
      if(measured_cycle_latency>=CYCLES_PER_BIN_AT_25MHZ)
        $fatal(1,"SYNC_SRAM_BIN_DEADLINE_FAIL cycles=%0d",measured_cycle_latency);
      input_words_this_bin=0;
    end
  end
  integer n_ready_stalls=0;
  neuroedge_speech_decoder dut(.*);

  // Stall logits for a few clocks to validate ready/valid stability.
  always @(negedge clk) begin
    if (!rst_n) logit_ready=0;
    else logit_ready=((($time/10)%9)>=2);
  end
  always @(posedge clk) begin
    if(rst_n && logit_valid && !logit_ready) n_ready_stalls=n_ready_stalls+1;
    if(rst_n && logit_valid && logit_ready) begin
      if(logit_frame !== 4*(seen/CLASSES))
        $fatal(1,"WRONG_FRAME got=%0d seen=%0d",logit_frame,seen);
      if(logit_index !== (seen%CLASSES))
        $fatal(1,"WRONG_CLASS got=%0d expected=%0d",logit_index,seen%CLASSES);
      if(logit_data !== $signed(expected_file[seen]))
        $fatal(1,"FULL_DECODER_MISMATCH output=%0d expected=%0d got=%0d",
                  seen,$signed(expected_file[seen]),logit_data);
      seen=seen+1;
    end
    if(rst_n && frame_done) frames_completed=frames_completed+1;
  end

  initial begin
    if (!$value$plusargs("WEIGHTS=%s",wpath)) $fatal(1,"MISSING_WEIGHTS");
    if (!$value$plusargs("BIASES=%s",bpath)) $fatal(1,"MISSING_BIASES");
    if (!$value$plusargs("SHIFTS=%s",spath)) $fatal(1,"MISSING_SHIFTS");
    if (!$value$plusargs("INPUTS=%s",xpath)) $fatal(1,"MISSING_INPUTS");
    if (!$value$plusargs("EXPECTED=%s",ypath)) $fatal(1,"MISSING_EXPECTED");
    $readmemh(wpath,weights_file);
    $readmemh(bpath,biases_file);
    $readmemh(spath,shifts_file);
    $readmemh(xpath,input_file);
    $readmemh(ypath,expected_file);
    repeat (4) @(negedge clk);
    rst_n=1;
    // RTL configuration ports, not hierarchical state injection.
    for(i=0;i<NWEIGHT;i=i+1) begin
      @(negedge clk);
      cfg_weight_addr=i[15:0];
      cfg_weight_data=$signed(weights_file[i]);
      cfg_weight_we=1;
    end
    @(negedge clk);cfg_weight_we=0;
    for(i=0;i<NMETA;i=i+1) begin
      @(negedge clk);
      cfg_meta_addr=i[8:0];
      cfg_meta_bias=$signed(biases_file[i]);
      cfg_meta_shift=shifts_file[i][4:0];
      cfg_meta_we=1;
    end
    @(negedge clk);cfg_meta_we=0;

    for(f=0;f<NF;f=f+1) begin
      for(c=0;c<INPUTS;c=c+1) begin
        @(negedge clk);
        if(in_ready !== 1'b1)
          $fatal(1,"INPUT_NOT_READY frame=%0d idx=%0d",f,c);
        in_data=$signed(input_file[f*INPUTS+c]);
        in_valid=1;
      end
      @(negedge clk);in_valid=0;
      // Must not send next frame until this inference finishes.
      // This is a streaming bin interface, not continuous 1-byte/cycle pipeline.
      wait(frame_done===1'b1);
      @(negedge clk);
    end
    repeat (4) @(negedge clk);
    if(frames_completed != NF) $fatal(1,"FRAME_COUNT mismatch %0d",frames_completed);
    if(seen != ((NF+3)/4)*CLASSES) $fatal(1,"LOGIT_COUNT mismatch %0d",seen);
    if(n_ready_stalls==0) $fatal(1,"NO_BACKPRESSURE_TESTED");
    $display("FULL_QUANTIZED_DECODER_RTL_PASS frames=%0d outputs=%0d weights=%0d stalls=%0d",
             NF,seen,NWEIGHT,n_ready_stalls);
    $display("SYNC_SRAM_FULL_DECODER_RTL_PASS frames=%0d logits=%0d worst_bin_cycles=%0d cycles_available=%0d", NF,seen,worst_cycle_latency,CYCLES_PER_BIN_AT_25MHZ);
    $finish;
  end
  initial begin
    #(NF*2000000);
    $fatal(1,"FULL_DECODER_TIMEOUT");
  end
endmodule
