module csr_regs(
  input logic clk, rst_n,
  input logic wr_en_i,
  input logic [7:0] addr_i,
  input logic [31:0] wdata_i,
  output logic [31:0] control_o,
  output logic [31:0] threshold_o
);
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin control_o<='0; threshold_o<=32'd32; end
    else if(wr_en_i) begin
      case(addr_i)
        8'h00: control_o <= wdata_i;
        8'h04: threshold_o <= wdata_i;
        default: ;
      endcase
    end
  end
endmodule
