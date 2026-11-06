module diag_ssm_cell #(
  parameter int W=16,
  parameter int FRAC=10
)(
  input logic clk, rst_n, valid_i,
  input logic signed [W-1:0] u_i,
  input logic signed [W-1:0] a_i,
  input logic signed [W-1:0] b_i,
  output logic valid_o,
  output logic signed [W-1:0] x_o
);
  logic signed [W-1:0] x_q;
  logic signed [2*W-1:0] ax,b_u;
  logic signed [2*W:0] sum;
  always_comb begin
    ax=$signed(a_i)*$signed(x_q);
    b_u=$signed(b_i)*$signed(u_i);
    sum=$signed(ax)+$signed(b_u);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin x_q<='0; valid_o<=1'b0; end
    else begin
      valid_o<=valid_i;
      if(valid_i) x_q <= sum >>> FRAC;
    end
  end
  assign x_o=x_q;
endmodule
