// Technology-specific NVM-CIM hard macro wrapper.
// This module intentionally contains no behavioral claim about RRAM/FeFET physics.
(* blackbox *)
module nvm_cim_blackbox #(
  parameter int IN_W=8,
  parameter int OUT_W=32,
  parameter int LANES=16
)(
  input  logic clk,
  input  logic req_valid_i,
  output logic req_ready_o,
  input  logic signed [IN_W-1:0] x_i [LANES],
  output logic rsp_valid_o,
  input  logic rsp_ready_i,
  output logic signed [OUT_W-1:0] y_o,
  output logic backend_error_o
);
endmodule
