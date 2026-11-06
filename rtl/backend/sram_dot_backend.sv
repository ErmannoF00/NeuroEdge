module sram_dot_backend #(
  parameter int IN_W=8,
  parameter int W_W=8,
  parameter int OUT_W=32,
  parameter int LANES=16,
  parameter logic signed [W_W-1:0] WEIGHTS [LANES] = '{default:'0}
)(
  input logic clk, rst_n,
  input logic req_valid_i,
  output logic req_ready_o,
  input logic signed [IN_W-1:0] x_i [LANES],
  output logic rsp_valid_o,
  input logic rsp_ready_i,
  output logic signed [OUT_W-1:0] y_o,
  output logic backend_error_o
);
  integer k;
  logic signed [OUT_W-1:0] sum_c;
  always_comb begin
    sum_c='0;
    for (k=0;k<LANES;k=k+1)
      sum_c = sum_c + $signed(x_i[k]) * $signed(WEIGHTS[k]);
  end
  assign req_ready_o = ~rsp_valid_o | rsp_ready_i;
  assign backend_error_o = 1'b0;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin rsp_valid_o<=1'b0; y_o<='0; end
    else if(req_ready_o) begin
      rsp_valid_o <= req_valid_i;
      if(req_valid_i) y_o <= sum_c;
    end
  end
endmodule
