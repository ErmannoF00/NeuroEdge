module lif_neuron #(
  parameter int W=16,
  parameter int signed BETA_NUM=15,
  parameter int signed BETA_DEN=16,
  parameter int signed THRESH=256
)(
  input logic clk, rst_n, valid_i,
  input logic signed [W-1:0] current_i,
  output logic spike_o,
  output logic signed [W-1:0] membrane_o
);
  logic signed [W-1:0] mem_q;
  logic signed [2*W-1:0] leak;
  logic signed [W:0] next_mem;
  always_comb begin
    leak = $signed(mem_q) * BETA_NUM;
    next_mem = $signed(leak / BETA_DEN) + $signed(current_i);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin mem_q<='0; spike_o<=1'b0; end
    else if(valid_i) begin
      if(next_mem >= THRESH) begin mem_q<='0; spike_o<=1'b1; end
      else begin mem_q<=next_mem[W-1:0]; spike_o<=1'b0; end
    end else spike_o<=1'b0;
  end
  assign membrane_o=mem_q;
endmodule
