`timescale 1ns/1ps
module tb_near_memory_dot;
  reg clk=0; always #5 clk=~clk;
  reg rst_n=0, cfg_valid=0, in_valid=0, out_ready=0;
  reg [2:0] cfg_addr=0;
  reg signed [7:0] cfg_weight=0, in_data=0;
  wire cfg_ready,in_ready,out_valid,busy;
  wire signed [31:0] out_sum;
  wire [31:0] accepted_inputs,accepted_writes;
  integer i, expected, value, weight;
  neuroedge_near_memory_dot #(.LENGTH(8),.ADDR_W(3)) dut (
    .clk(clk),.rst_n(rst_n),.cfg_valid(cfg_valid),.cfg_ready(cfg_ready),
    .cfg_addr(cfg_addr),.cfg_weight(cfg_weight),.in_valid(in_valid),
    .in_ready(in_ready),.in_data(in_data),.out_valid(out_valid),
    .out_ready(out_ready),.out_sum(out_sum),.busy(busy),
    .accepted_inputs(accepted_inputs),.accepted_writes(accepted_writes));
  task write_weight(input integer adr,input integer w);
    begin
      @(negedge clk); cfg_addr=adr; cfg_weight=w; cfg_valid=1;
      if (!cfg_ready) $fatal(1,"cfg should be ready at row boundary");
      @(negedge clk); cfg_valid=0;
    end
  endtask
  task push(input integer v);
    begin
      @(negedge clk); in_data=v; in_valid=1;
      while (!in_ready) @(negedge clk);
      @(negedge clk); in_valid=0;
    end
  endtask
  initial begin
    repeat(3) @(negedge clk); rst_n=1;
    // Alternating extreme weights, positive/negative inputs.
    for(i=0;i<8;i=i+1) write_weight(i,(i%2==0)?127:-128);
    expected=0;
    for(i=0;i<8;i=i+1) begin
      value=i*19-70; weight=(i%2==0)?127:-128;
      expected=expected+value*weight;
      push(value);
      // A config write must never be granted during a partial row.
      if (i==2) begin
        @(negedge clk); cfg_addr=0; cfg_weight=44; cfg_valid=1;
        if(cfg_ready) $fatal(1,"unsafe config access accepted");
        @(negedge clk); cfg_valid=0;
      end
    end
    wait(out_valid);
    if(out_sum !== expected) $fatal(1,"dot mismatch observed=%0d expected=%0d",out_sum,expected);
    repeat(5) begin
      @(negedge clk); if (!out_valid || out_sum !== expected)
        $fatal(1,"result not held under backpressure");
    end
    out_ready=1;
    @(negedge clk); out_ready=0;
    repeat(2) @(negedge clk);
    if(accepted_inputs!==8 || accepted_writes!==8 || busy)
      $fatal(1,"accounting/cleanup failure in=%0d writes=%0d busy=%b",accepted_inputs,accepted_writes,busy);
    $display("NEUROEDGE_NEAR_MEMORY_DOT_RTL_PASS sum=%0d inputs=%0d writes=%0d",expected,accepted_inputs,accepted_writes);
    $finish;
  end
  initial begin #200000; $fatal(1,"timeout"); end
endmodule
