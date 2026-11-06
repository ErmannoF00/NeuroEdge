module nmc_tile #(
  parameter int W=8,
  parameter int ACC_W=32,
  parameter int LANES=16
)(
  input  logic signed [W-1:0] act_i [LANES],
  input  logic signed [W-1:0] wgt_i [LANES],
  output logic signed [ACC_W-1:0] sum_o
);
  integer i;
  logic signed [ACC_W-1:0] tmp;
  always_comb begin
    tmp='0;
    for (i=0;i<LANES;i=i+1) tmp = tmp + $signed(act_i[i])*$signed(wgt_i[i]);
    sum_o=tmp;
  end
endmodule
