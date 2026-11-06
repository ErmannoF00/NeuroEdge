`timescale 1ns/1ps
module tb_sync_sram_1rw;
  reg clk=0;
  always #5 clk=~clk;
  reg en=0, we=0;
  reg [3:0] addr=0;
  reg [7:0] wdata=0;
  wire [7:0] rdata;
  integer i;
  neuroedge_sram_1rw #(.WIDTH(8),.DEPTH(16),.ADDR_W(4)) dut (
    .clk(clk),.en(en),.we(we),.addr(addr),.wdata(wdata),.rdata(rdata)
  );
  initial begin
    for(i=0;i<16;i=i+1) begin
      @(negedge clk);
      en=1; we=1; addr=i[3:0]; wdata=8'(i*9+3);
    end
    @(negedge clk);
    en=0; we=0;
    for(i=0;i<16;i=i+1) begin
      @(negedge clk);
      en=1; we=0; addr=i[3:0];
      @(posedge clk); #1;
      if(rdata !== 8'(i*9+3))
        $fatal(1,"SRAM_MISMATCH index=%0d got=%0d expected=%0d",i,rdata,8'(i*9+3));
    end
    @(negedge clk);
    en=0; we=0; addr=0;
    repeat(3) begin
      @(posedge clk); #1;
      if(rdata !== 8'(15*9+3)) $fatal(1,"SRAM_HOLD_FAIL");
    end
    $display("SYNC_SRAM_UNIT_PASS reads=16 writes=16 hold=3");
    $finish;
  end
  initial begin
    #10000;
    $fatal(1,"SRAM_UNIT_TIMEOUT");
  end
endmodule
