`timescale 1ns/1ps
module tb_budget_controller;
 reg clk=0; always #5 clk=~clk;
 reg rst_n=0, sample_valid=0;
 reg [15:0] drift_score=0;
 wire drift_active, update_request;
 wire [15:0] writes_in_window;
 wire [31:0] observed_bins,suppressed_updates;
 neuroedge_budget_controller dut(
  .clk(clk),.rst_n(rst_n),.sample_valid(sample_valid),.drift_score(drift_score),
  .drift_active(drift_active),.update_request(update_request),
  .writes_in_window(writes_in_window),.observed_bins(observed_bins),
  .suppressed_updates(suppressed_updates));
 integer fd, ret, score, active, req, writes, suppressed, n=0;
 reg [1023:0] filename;
 initial begin
  if (!$value$plusargs("VECTORS=%s",filename)) $fatal(1,"Missing +VECTORS");
  fd=$fopen(filename,"r");
  if(fd==0) $fatal(1,"Unable to open vectors");
  repeat(3) @(negedge clk);
  rst_n=1;
  while (!$feof(fd)) begin
    ret=$fscanf(fd,"%d %d %d %d %d\n",score,active,req,writes,suppressed);
    if(ret==5) begin
      @(negedge clk); drift_score=score; sample_valid=1;
      @(posedge clk); #1;
      if (drift_active !== (active!=0) || update_request !== (req!=0) ||
          writes_in_window !== writes || suppressed_updates !== suppressed)
         $fatal(1,"MISMATCH bin=%0d score=%0d expected=%0d,%0d,%0d,%0d got=%0d,%0d,%0d,%0d", n, score,active,req,writes,suppressed,drift_active,update_request,writes_in_window,suppressed_updates);
      @(negedge clk); sample_valid=0;
      n=n+1;
    end else if(ret!=-1) $fatal(1,"Malformed vector row");
  end
  if(observed_bins !== n) $fatal(1,"SAMPLE_COUNT_FAIL");
  $display("BUDGET_CONTROLLER_RTL_PASS bins=%0d suppressed=%0d",n,suppressed_updates);
  $finish;
 end
endmodule
