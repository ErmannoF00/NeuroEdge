`timescale 1ns/1ps
module tb_neuroedge_mmio;
  parameter integer NF=8;
  localparam integer NWEIGHT=44224, NMETA=361, INPUTS=512, CLASSES=41;
  reg clk=0;
  always #5 clk=~clk;
  reg rst_n=0;
  reg req_valid=0,req_write=0;
  reg [7:0] req_addr=0;
  reg [31:0] req_wdata=0;
  wire req_ready;
  wire [31:0] req_rdata;
  reg [7:0] weights_file[0:NWEIGHT-1];
  reg [31:0] biases_file[0:NMETA-1];
  reg [7:0] shifts_file[0:NMETA-1];
  reg [7:0] input_file[0:NF*INPUTS-1];
  reg [15:0] expected_file[0:NF/4*CLASSES-1];
  string wpath,bpath,spath,xpath,ypath;
  reg [31:0] value, score, frameval;
  integer i,f,c,k,tries,seen=0;
  neuroedge_mmio_wrapper dut(.*);

  task automatic write32(input [7:0] addr, input [31:0] dat);
    integer timeout;
    begin
      @(negedge clk);
      req_addr=addr;req_wdata=dat;req_valid=1;req_write=1;
      #1;
      timeout=0;
      while(req_ready !== 1'b1) begin
        @(negedge clk);
        timeout=timeout+1;
        if(timeout>100000) $fatal(1,"MMIO_WRITE_TIMEOUT addr=%h",addr);
      end
      @(posedge clk);
      @(negedge clk);req_valid=0;req_write=0;
    end
  endtask
  task automatic read32(input [7:0] addr,output [31:0] dat);
    integer timeout;
    begin
      @(negedge clk);
      req_addr=addr;req_valid=1;req_write=0;
      #1;
      timeout=0;
      while(req_ready !== 1'b1) begin
        @(negedge clk);
        timeout=timeout+1;
        if(timeout>100000) $fatal(1,"MMIO_READ_TIMEOUT addr=%h",addr);
      end
      dat=req_rdata;
      @(posedge clk);
      @(negedge clk);req_valid=0;
    end
  endtask

  initial begin
    if (!$value$plusargs("WEIGHTS=%s",wpath)) $fatal(1,"NO_WEIGHTS");
    if (!$value$plusargs("BIASES=%s",bpath)) $fatal(1,"NO_BIASES");
    if (!$value$plusargs("SHIFTS=%s",spath)) $fatal(1,"NO_SHIFTS");
    if (!$value$plusargs("INPUTS=%s",xpath)) $fatal(1,"NO_INPUTS");
    if (!$value$plusargs("EXPECTED=%s",ypath)) $fatal(1,"NO_EXPECTED");
    $readmemh(wpath,weights_file);
    $readmemh(bpath,biases_file);
    $readmemh(spath,shifts_file);
    $readmemh(xpath,input_file);
    $readmemh(ypath,expected_file);
    repeat(4) @(negedge clk);
    rst_n=1;
    read32(8'h00,value);
    if(value!==32'h4e453139) $fatal(1,"BAD_MMIO_ID");
    // Before config commit no neural sample may be accepted.
    @(negedge clk);
    req_valid=1;req_write=1;req_addr=8'h20;req_wdata=32'd0;
    #1;
    if(req_ready!==1'b0) $fatal(1,"UNCOMMITTED_INPUT_ACCEPTED");
    @(negedge clk);req_valid=0;req_write=0;

    for(i=0;i<NWEIGHT;i=i+1) write32(8'h0c,{24'd0,weights_file[i]});
    for(i=0;i<NMETA;i=i+1) begin
      write32(8'h14,biases_file[i]);
      write32(8'h18,{24'd0,shifts_file[i]});
    end
    read32(8'h08,value);
    if(value!==NWEIGHT) $fatal(1,"WEIGHT_PTR_BAD %d",value);
    read32(8'h10,value);
    if(value!==NMETA) $fatal(1,"META_PTR_BAD %d",value);
    write32(8'h1c,1);
    read32(8'h04,value);
    if(value[0]!==1'b1 || value[3]!==1'b0) $fatal(1,"CONFIG_COMMIT_FAILED %h",value);

    for(f=0;f<NF;f=f+1) begin
      for(c=0;c<INPUTS;c=c+1) write32(8'h20,{24'd0,input_file[f*INPUTS+c]});
      if((f%4)==0) begin
        for(k=0;k<CLASSES;k=k+1) begin
          tries=0;
          begin: WAIT_LOGIT
            while(1) begin
              read32(8'h04,value);
              if(value[2]) disable WAIT_LOGIT;
              repeat(64) @(negedge clk);
              tries=tries+1;
              if(tries>10000) $fatal(1,"RESULT_TIMEOUT f=%0d k=%0d",f,k);
            end
          end
          read32(8'h28,frameval);
          read32(8'h24,score);
          if(frameval!==f) $fatal(1,"FRAME_ID_MISMATCH expected=%0d got=%0d",f,frameval);
          if(score[21:16]!==k[5:0]) $fatal(1,"CLASS_INDEX_MISMATCH f=%0d k=%0d got=%0d",f,k,score[21:16]);
          if(score[15:0]!==expected_file[seen])
            $fatal(1,"MMIO_LOGIT_MISMATCH index=%0d got=%h expected=%h",seen,score[15:0],expected_file[seen]);
          seen=seen+1;
        end
      end
      tries=0;
      begin: WAIT_FRAME_COMPLETE
        while(1) begin
          read32(8'h2c,value);
          if(value==f+1) disable WAIT_FRAME_COMPLETE;
          repeat(16) @(negedge clk);
          tries=tries+1;
          if(tries>20000) $fatal(1,"FRAME_DONE_TIMEOUT f=%0d completed=%0d",f,value);
        end
      end
    end
    if(seen != NF/4*CLASSES) $fatal(1,"OUTPUT_COUNT_BAD got=%0d",seen);
    read32(8'h04,value);
    if(value[3]) $fatal(1,"STICKY_MMIO_FAULT");
    $display("NEUROEDGE_MMIO_RTL_PASS frames=%0d logits=%0d writes=%0d",NF,seen,NWEIGHT);
    $finish;
  end
  initial begin
    #(NF*3000000);
    $fatal(1,"MMIO_TEST_TIMEOUT");
  end
endmodule
