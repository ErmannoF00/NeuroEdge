module mac_lane #(
  parameter int W=8,
  parameter int ACC_W=32
)(
  input  logic signed [W-1:0] a_i,
  input  logic signed [W-1:0] w_i,
  input  logic signed [ACC_W-1:0] acc_i,
  output logic signed [ACC_W-1:0] acc_o
);
  logic signed [2*W-1:0] prod;
  always_comb begin
    prod  = $signed(a_i) * $signed(w_i);
    acc_o = $signed(acc_i) + $signed(prod);
  end
endmodule
