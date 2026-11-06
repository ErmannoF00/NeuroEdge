interface neuro_backend_if #(
  parameter int IN_W=8,
  parameter int OUT_W=32,
  parameter int LANES=16
)(input logic clk);
  logic req_valid, req_ready;
  logic signed [IN_W-1:0] x [LANES];
  logic rsp_valid, rsp_ready;
  logic signed [OUT_W-1:0] y;
  logic backend_error;

  modport host(output req_valid, x, rsp_ready,
               input req_ready, rsp_valid, y, backend_error);
  modport backend(input req_valid, x, rsp_ready,
                  output req_ready, rsp_valid, y, backend_error);
endinterface
