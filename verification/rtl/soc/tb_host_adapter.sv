`timescale 1ns/1ps
// Full 32-bin bus-driven decoder test. Synthetic or local fixture supplied by
// caller. Zero/committed offsets are compared to independent Python goldens.
// No hierarchical writes into the decoder or calibration bank.
module tb_host_adapter;
    parameter integer NF=32;
    parameter integer CASE_ID=0; // 0 denied mid-bin, 1 accepted pre-bin offset
    localparam integer NWEIGHTS=44224, NMETA=361, INPUTS=512, CLASSES=41;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg req_valid=0,req_write=0;
    reg [7:0] req_addr=0;
    reg [31:0] req_wdata=0;
    wire req_ready,irq;
    wire [31:0] req_rdata;
    reg [7:0] weight_mem [0:NWEIGHTS-1];
    reg [31:0] bias_mem [0:NMETA-1];
    reg [7:0] shift_mem [0:NMETA-1];
    reg [7:0] input_mem [0:NF*INPUTS-1];
    reg [15:0] expected_mem [0:NF/4*CLASSES-1];
    reg [15:0] baseline_mem [0:NF/4*CLASSES-1];
    string wp,bp,sp,xp,yp,zp;
    integer channel,offset_value;
    integer i,f,c,seen=0,changed=0,stalls=0,iterations=0;
    integer worst_core_latency=0;
    integer first_backpressure=0;
    integer clock_cycles=0,bus_stall_cycles=0,core_output_blocked=0;
    always @(posedge clk) if(rst_n) begin
        clock_cycles=clock_cycles+1;
        if(req_valid && !req_ready) bus_stall_cycles=bus_stall_cycles+1;
        if(dut.logit_valid && !dut.u_pipeline.logit_ready)
            core_output_blocked=core_output_blocked+1;
    end
    reg [31:0] value,word_frame,word_result;

    neuroedge_host_adapter #(.INPUT_DEPTH(4),.OUTPUT_DEPTH(4)) dut (.*);

    task automatic write_reg(input [7:0] addr,input [31:0] data);
       integer attempts;
       begin
         @(negedge clk);
         req_addr=addr;req_wdata=data;req_write=1;req_valid=1;
         // req_ready is combinational: allow it to settle before testing.
         #1;
         attempts=0;
         while(req_ready!==1'b1) begin
            @(negedge clk);
            #1;
            attempts=attempts+1;
            if(attempts>350000) $fatal(1,"BUS_WRITE_TIMEOUT addr=%h",addr);
         end
         // Exactly one accepted posedge; deassert before any second posedge.
         @(posedge clk);
         @(negedge clk);req_valid=0;req_write=0;
       end
    endtask
    task automatic read_reg(input [7:0] addr,output reg [31:0] data);
       integer attempts;
       begin
         @(negedge clk);
         req_addr=addr;req_write=0;req_valid=1;
         #1;
         attempts=0;
         while(req_ready!==1'b1) begin
            @(negedge clk);
            #1;
            attempts=attempts+1;
            if(attempts>350000) $fatal(1,"BUS_READ_TIMEOUT addr=%h",addr);
         end
         // Capture the combinational read value before its destructive edge.
         data=req_rdata;
         @(posedge clk);
         @(negedge clk);req_valid=0;
       end
    endtask
    task automatic consume_one;
       begin
         if(!irq) $fatal(1,"RESULT_IRQ_NOT_ASSERTED");
         read_reg(8'h28,word_frame); // peek frame BEFORE destructive dequeue
         read_reg(8'h24,word_result);
         if(seen>=NF/4*CLASSES) $fatal(1,"EXTRA_OUTPUT");
         if(word_frame !== (4*(seen/CLASSES)))
             $fatal(1,"WRONG_FRAME idx=%0d expected=%0d got=%0d",seen,4*(seen/CLASSES),word_frame);
         if(word_result[21:16] !== (seen%CLASSES))
             $fatal(1,"WRONG_CLASS idx=%0d got=%0d",seen,word_result[21:16]);
         if(word_result[15:0] !== expected_mem[seen])
             $fatal(1,"HOST_LOGIT_MISMATCH idx=%0d expected=%d got=%d",
                    seen,$signed(expected_mem[seen]),$signed(word_result[15:0]));
         if(word_result[15:0] !== baseline_mem[seen]) changed=changed+1;
         seen=seen+1;
       end
    endtask
    task automatic stage_one(input integer ch, input integer v);
       begin
         write_reg(8'h30,ch);
         write_reg(8'h34,v);
         read_reg(8'h3c,value);
         if(value[0]!==1'b1) $fatal(1,"HOST_STAGING_MISSING");
       end
    endtask
    initial begin
      if(!$value$plusargs("WEIGHTS=%s",wp) || !$value$plusargs("BIASES=%s",bp) ||
         !$value$plusargs("SHIFTS=%s",sp) || !$value$plusargs("INPUTS=%s",xp) ||
         !$value$plusargs("EXPECTED=%s",yp) || !$value$plusargs("BASELINE=%s",zp))
           $fatal(1,"HOST_FIXTURE_PATH_MISSING");
      if(CASE_ID==1) begin
          if(!$value$plusargs("CHANNEL=%d",channel) ||
             !$value$plusargs("OFFSET=%d",offset_value))
              $fatal(1,"HOST_COMMIT_PARAMETERS_MISSING");
      end
      $readmemh(wp,weight_mem); $readmemh(bp,bias_mem);
      $readmemh(sp,shift_mem); $readmemh(xp,input_mem);
      $readmemh(yp,expected_mem); $readmemh(zp,baseline_mem);
      repeat(4) @(negedge clk); rst_n=1;
      read_reg(8'h00,value);
      if(value!==32'h4e454443) $fatal(1,"INVALID_HOST_ID");
      // Illegal early seal must NOT configure anything.
      write_reg(8'h1c,1);read_reg(8'h60,value);
      if((value & 4)==0) $fatal(1,"EARLY_SEAL_NOT_REJECTED");
      write_reg(8'h60,32'hffffffff);
      for(i=0;i<NWEIGHTS;i=i+1) write_reg(8'h0c,{24'd0,weight_mem[i]});
      for(i=0;i<NMETA;i=i+1) begin
          write_reg(8'h14,bias_mem[i]);
          write_reg(8'h18,{24'd0,shift_mem[i]});
      end
      read_reg(8'h08,value);
      if(value!==NWEIGHTS) $fatal(1,"WEIGHT_COUNT_MISMATCH");
      read_reg(8'h10,value);
      if(value!==NMETA) $fatal(1,"META_COUNT_MISMATCH");
      write_reg(8'h1c,1);
      read_reg(8'h04,value);
      if(!value[0]) $fatal(1,"CONFIGURATION_SEAL_MISSING");
      // Program an output/fault/completion interrupt policy.
      write_reg(8'h54,7);
      // After sealing, configuration is immutable; failed attempts raise fault.
      write_reg(8'h0c,32'd17);
      read_reg(8'h60,value);
      if((value&2)==0) $fatal(1,"SEALED_WEIGHT_MUTATION_NOT_FAULTED");
      write_reg(8'h60,32'hffffffff);
      if(CASE_ID==1) begin
          stage_one(channel,offset_value);
          write_reg(8'h38,1);
          repeat(2) @(negedge clk);
          read_reg(8'h40,value);
          if(value!==1) $fatal(1,"COMMIT_NOT_COUNTED");
          read_reg(8'h48,value);
          if(value!==1) $fatal(1,"COMMIT_REQUEST_MISSING");
      end
      // An empty RESULT read must not falsely transfer a FIFO entry.
      @(negedge clk);req_valid=1;req_write=0;req_addr=8'h24;
      #1;if(req_ready!==0) $fatal(1,"EMPTY_RESULT_NOT_STALLED");
      @(negedge clk);req_valid=0;

      for(f=0;f<NF;f=f+1) begin
        for(c=0;c<INPUTS;c=c+1) begin
          write_reg(8'h20,{24'd0,input_mem[f*INPUTS+c]});
          if(CASE_ID==0 && f==0 && c==127) begin
             // The bridge owns a partial bin even if the decoder reports idle.
             stage_one(7,120);
             write_reg(8'h38,1); // must be rejected immediately, no retry
             repeat(2) @(negedge clk);
             read_reg(8'h44,value);

             $display(
               "HOST_AUTH_DEBUG read=%h rejected=%0d accepted=%0d writes=%0d pending=%b bin_in_flight=%b feature_index=%0d busy=%b fifo=%0d",
               value,
               dut.rejected_requests,
               dut.accepted_requests,
               dut.accepted_writes,
               dut.pending,
               dut.bin_in_flight,
               dut.feature_index,
               dut.busy,
               dut.in_count
             );
             if(value!==1) $fatal(1,"MID_BIN_AUTHORIZATION_NOT_REJECTED");
             read_reg(8'h40,value);
             if(value!==0) $fatal(1,"MID_BIN_WRITE_ACCEPTED");
             write_reg(8'h38,2); // abort pending candidate
             read_reg(8'h3c,value);
             if(value[0]) $fatal(1,"ABORT_DID_NOT_CLEAR_PENDING");
          end
        end
        if(f==0) begin
           // Force actual output FIFO saturation before reading, exercising
           // the pipeline's registered logit_valid / logit_ready stalls.
           iterations=0;
           while(dut.out_count!=4) begin
              @(negedge clk);iterations=iterations+1;
              if(iterations>300000) $fatal(1,"NO_OUTPUT_BACKPRESSURE");
           end
           first_backpressure=1;
           // FIFO occupancy alone does not prove producer backpressure:
           // wait until a fifth logit is presented while the FIFO is full.
           // The decoder's per-class MAC takes more than 20 clocks.
           iterations=0;
           while(!(dut.logit_valid && !dut.u_pipeline.logit_ready)) begin
              @(negedge clk);
              iterations=iterations+1;
              if(iterations>300000)
                 $fatal(1,"CORE_OUTPUT_BACKPRESSURE_NOT_REACHED");
           end
           if(dut.out_count!=4 || dut.u_pipeline.logit_ready!==1'b0)
              $fatal(1,"CORE_OUTPUT_BACKPRESSURE_INVALID_STATE");
           begin : hold_stalled_result
              reg [15:0] held_data;
              reg [5:0] held_index;
              reg [31:0] held_frame;
              held_data=dut.logit_data;
              held_index=dut.logit_index;
              held_frame=dut.logit_frame;
              repeat(20) @(negedge clk);
              if(dut.out_count!=4 || dut.logit_valid!==1'b1 ||
                 dut.u_pipeline.logit_ready!==1'b0 ||
                 dut.logit_data!==held_data ||
                 dut.logit_index!==held_index ||
                 dut.logit_frame!==held_frame)
                  $fatal(1,"STALLED_CORE_LOGIT_NOT_STABLE");
           end
           $display("HOST_CORE_BACKPRESSURE_OBSERVED blocked_cycles=%0d",core_output_blocked);
           read_reg(8'h28,word_frame);
           repeat(10) @(negedge clk);
           read_reg(8'h28,value);
           if(value!==word_frame || dut.out_count!=4)
               $fatal(1,"NON_DESTRUCTIVE_RESULT_PEEK_FAILED");
        end
        iterations=0;
        // Drain outputs while the decoder is computing; waiting only for
        // frame_done would deadlock when the output FIFO is full.
        begin : drain_bin
          reg finished;
          finished=0;
          while(!finished) begin
            read_reg(8'h04,value);
            if(value[2]) begin
               consume_one();
            end else begin
               read_reg(8'h2c,value);
               if(value>=f+1) finished=1;
            end
            iterations=iterations+1;
            if(iterations>350000) $fatal(1,"BIN_DRAIN_TIMEOUT f=%0d",f);
          end
        end
        read_reg(8'h68,value);
        if(value==0 || value>=500000) $fatal(1,"HOST_BIN_LATENCY_FAIL bin=%0d cycles=%0d",f,value);
        if(value>worst_core_latency) worst_core_latency=value;
      end
      // Consume any logits remaining from the last completed bin.
      repeat(5) @(negedge clk);
      read_reg(8'h04,value);
      while(value[2]) begin
          consume_one();
          read_reg(8'h04,value);
      end
      if(core_output_blocked==0) $fatal(1,"NO_ACTUAL_CORE_BACKPRESSURE");
      if(seen!==NF/4*CLASSES || first_backpressure==0)
          $fatal(1,"HOST_OUTPUT_COUNT_MISMATCH seen=%0d",seen);
      read_reg(8'h4c,value);
      if(value!==NF*INPUTS) $fatal(1,"HOST_INPUT_COUNT_MISMATCH got=%0d",value);
      read_reg(8'h50,value);
      if(value!==seen) $fatal(1,"HOST_POP_COUNT_MISMATCH");
      read_reg(8'h2c,value);
      if(value!==NF) $fatal(1,"HOST_BIN_COUNT_MISMATCH");
      read_reg(8'h68,value);
      if(value>=500000 || value==0) $fatal(1,"HOST_LATENCY_OUT_OF_RANGE");
      if(CASE_ID==0) begin
          if(changed!=0) $fatal(1,"HOST_BASELINE_ALTERED");
          read_reg(8'h40,value);
          if(value!=0) $fatal(1,"UNEXPECTED_OFFSET_WRITE");
          $display("HOST_ZERO_OFFSET_EXACT_PASS logits=%0d inputs=%0d",seen,NF*INPUTS);
          $display("HOST_UNSAFE_UPDATE_REJECTED_PASS");
      end else begin
          if(changed==0) $fatal(1,"COMMIT_DID_NOT_CHANGE_OUTPUT");
          $display("HOST_COMMITTED_OFFSET_EXACT_PASS logits=%0d changed=%0d",seen,changed);
      end
      read_reg(8'h60,value);
      if(CASE_ID==0 && (value&8)==0) $fatal(1,"UNSAFE_REQUEST_FAULT_ABSENT");
      write_reg(8'h60,32'hffffffff);
      write_reg(8'h5c,4);
      read_reg(8'h60,value);
      if(value!=0) $fatal(1,"FAULT_CLEAR_FAILURE");
      read_reg(8'h58,value);
      if(value[2] || value[1] || value[0]) $fatal(1,"IRQ_ACK_FAILURE");
      if(irq!==0) $fatal(1,"IRQ_STUCK");
      @(negedge clk);rst_n=0;
      repeat(3) @(negedge clk);rst_n=1;
      read_reg(8'h04,value);
      if(value!==0) $fatal(1,"HOST_RESET_STATUS_NOT_ZERO");
      read_reg(8'h08,value);
      if(value!==0) $fatal(1,"HOST_RESET_WEIGHT_PTR_NOT_ZERO");
      read_reg(8'h64,value);
      if(value!==0) $fatal(1,"HOST_RESET_FIFO_NOT_ZERO");
      $display("HOST_BUS_ACCOUNTING cycles=%0d request_stall_cycles=%0d logit_backpressure_cycles=%0d worst_bin_cycles=%0d",clock_cycles,bus_stall_cycles,core_output_blocked,worst_core_latency);
      $display("NEUROEDGE_HOST_ADAPTER_RTL_PASS bins=%0d logits=%0d input_fifo=4 output_fifo=4",NF,seen);
      $finish;
    end
    initial begin
      #120000000;
      $fatal(1,"HOST_GLOBAL_TIMEOUT seen=%0d",seen);
    end
endmodule
