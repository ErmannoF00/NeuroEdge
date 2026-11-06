// One-entry staged update bank, 512 signed INT16 channel offsets.
// existing: request is authorization, not learning; decoded-frame IDLE is a safety gate.
// Latency: registered read_data, 1 cycle, read-old-data on same-address commit.
// Reset clears valid flags, NOT SRAM contents. Missing entries read zero.
// No energy, NVM endurance, or complete adaptation claim.
module neuroedge_offset_bank #(
    parameter integer CHANNELS = 512,
    parameter integer ADDR_W = 9
)(
    input wire clk,
    input wire rst_n,
    input wire stage_valid,
    output wire stage_ready,
    input wire [ADDR_W-1:0] stage_channel,
    input wire signed [15:0] stage_value,
    input wire abort_stage,
    input wire update_request,
    input wire decoder_idle,
    input wire read_enable,
    input wire [ADDR_W-1:0] read_channel,
    output reg signed [15:0] read_data,
    output reg pending,
    output reg write_accepted,
    output reg [31:0] accepted_writes,
    output reg [31:0] rejected_requests,
    output reg [31:0] stages_accepted,
    output reg [31:0] aborted_stages
);
    reg signed [15:0] active [0:CHANNELS-1];
    reg [CHANNELS-1:0] valid_bits;
    reg [ADDR_W-1:0] pending_channel;
    reg signed [15:0] pending_value;
    assign stage_ready = !pending && !abort_stage;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending <= 1'b0;
            pending_channel <= 0;
            pending_value <= 0;
            write_accepted <= 1'b0;
            accepted_writes <= 0;
            rejected_requests <= 0;
            stages_accepted <= 0;
            aborted_stages <= 0;
            valid_bits <= 0;
            read_data <= 0;
        end else begin
            write_accepted <= 1'b0;
            if (read_enable)
                read_data <= valid_bits[read_channel] ? active[read_channel] : 16'sd0;
            if (abort_stage) begin
                if (pending) aborted_stages <= aborted_stages + 1;
                pending <= 1'b0;
            end else begin
                if (stage_valid && stage_ready) begin
                    pending <= 1'b1;
                    pending_channel <= stage_channel;
                    pending_value <= stage_value;
                    stages_accepted <= stages_accepted + 1;
                end
                if (update_request) begin
                    if (pending && decoder_idle) begin
                        active[pending_channel] <= pending_value;
                        valid_bits[pending_channel] <= 1'b1;
                        accepted_writes <= accepted_writes + 1;
                        write_accepted <= 1'b1;
                        pending <= 1'b0;
                    end else begin
                        rejected_requests <= rejected_requests + 1;
                    end
                end
            end
        end
    end
endmodule
