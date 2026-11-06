`timescale 1ns/1ps
module tb_sram_dot_backend;
  localparam int LANES=4;
  logic clk=0, rst_n=0, req_valid, req_ready, rsp_valid, rsp_ready=1, err;
  logic signed [7:0] x[LANES];
  logic signed [31:0] y;
  localparam logic signed [7:0] W[LANES]='{8'sd1,8'sd2,-8'sd1,8'sd3};
  always #5 clk=~clk;
  sram_dot_backend #(.LANES(LANES),.WEIGHTS(W)) dut(.*,
    .req_valid_i(req_valid),.req_ready_o(req_ready),.x_i(x),
    .rsp_valid_o(rsp_valid),.rsp_ready_i(rsp_ready),.y_o(y),.backend_error_o(err));
  initial begin
    req_valid=0; x[0]=1; x[1]=2; x[2]=3; x[3]=4;
    #12; rst_n=1; #10; req_valid=1; #10; req_valid=0;
    wait(rsp_valid); if(y!==14) $fatal(1,"expected 14 got %0d",y);
    #10; $finish;
  end
endmodule
