`timescale 1ns/1ps
// FPGA/ASIC-synthesis memory inference template with single physical 1RW port.
// A read is synchronous and becomes visible after a clock edge; no write-first
// guarantee is provided. Host configuration and inference never overlap.
// IMPORTANT: technology SRAM macro, MBIST, PVT timing and read power NOT supplied.
module neuroedge_sram_1rw #(
    parameter integer WIDTH=8,
    parameter integer DEPTH=44224,
    parameter integer ADDR_W=16
) (
    input logic clk,
    input logic en,
    input logic we,
    input logic [ADDR_W-1:0] addr,
    input logic [WIDTH-1:0] wdata,
    output logic [WIDTH-1:0] rdata
);
  (* ram_style="block" *) logic [WIDTH-1:0] mem [0:DEPTH-1];
  always_ff @(posedge clk) begin
    if(en) begin
      if(we) mem[addr] <= wdata;
      else rdata <= mem[addr];
    end
  end
endmodule
