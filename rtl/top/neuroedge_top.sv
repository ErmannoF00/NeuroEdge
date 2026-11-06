module neuroedge_top #(
  parameter int W=16
)(
  input logic clk, rst_n,
  input logic sample_valid_i,
  input logic signed [W-1:0] sample_i,
  output logic event_valid_o,
  output logic signed [1:0] event_o
);
  delta_encoder #(.W(W),.THRESH(32)) u_delta(
    .clk(clk),.rst_n(rst_n),.valid_i(sample_valid_i),.sample_i(sample_i),
    .valid_o(event_valid_o),.event_o(event_o)
  );
endmodule
