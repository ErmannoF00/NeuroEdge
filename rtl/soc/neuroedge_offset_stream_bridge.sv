`timescale 1ns/1ps
// existing: 512 signed-Q4 feature words per bin, strictly serialized registered
// coefficient read followed by ready/valid core handoff. No offset update here.
// Timing: external handshake requests SRAM at edge N; RAM output registered N;
// B_WAIT at N+1 latches nothing and enters B_SEND; core consumes >= edge N+2.
module neuroedge_offset_stream_bridge (
    input logic clk, rst_n,
    input logic feature_valid,
    output logic feature_ready,
    input logic signed [7:0] feature_data,
    output logic offset_read_enable,
    output logic [8:0] offset_read_channel,
    input logic signed [15:0] offset_read_data,
    output logic core_in_valid,
    input logic core_in_ready,
    output logic signed [7:0] core_in_data,
    input logic core_frame_done,
    input logic allow_features,
    output logic bin_in_flight,
    output logic bridge_empty,
    output logic [8:0] feature_index,
    output logic [31:0] bin_latency_cycles,
    output logic [31:0] bins_completed
);
    typedef enum logic [1:0] {B_EMPTY, B_WAIT, B_SEND, B_FINISH} state_t;
    state_t state;
    logic signed [7:0] raw_latched;
    logic [31:0] cycle_counter, bin_start_cycle;
    logic signed [16:0] signed_sum;

    always_comb begin
        feature_ready = rst_n && (state == B_EMPTY) && allow_features && core_in_ready;
        bridge_empty = (state == B_EMPTY);
        offset_read_enable = feature_valid && feature_ready;
        offset_read_channel = feature_index;
        core_in_valid = (state == B_SEND);
        signed_sum = $signed({{9{raw_latched[7]}}, raw_latched}) +
                     $signed({offset_read_data[15],offset_read_data});
        if (signed_sum > 17'sd127) core_in_data = 8'sd127;
        else if (signed_sum < -17'sd128) core_in_data = 8'sh80;
        else core_in_data = signed_sum[7:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= B_EMPTY;
            feature_index <= 0;
            bin_in_flight <= 0;
            raw_latched <= 0;
            bin_latency_cycles <= 0;
            bins_completed <= 0;
            cycle_counter <= 0;
            bin_start_cycle <= 0;
        end else begin
            cycle_counter <= cycle_counter + 1'b1;
            case (state)
                B_EMPTY: if (offset_read_enable) begin
                    raw_latched <= feature_data;
                    if (feature_index == 0) begin
                        bin_in_flight <= 1;
                        bin_start_cycle <= cycle_counter;
                    end
                    state <= B_WAIT;
                end
                B_WAIT: state <= B_SEND;
                B_SEND: if (core_in_valid && core_in_ready) begin
                    if (feature_index == 9'd511) begin
                        state <= B_FINISH;
                    end else begin
                        feature_index <= feature_index + 1'b1;
                        state <= B_EMPTY;
                    end
                end
                B_FINISH: if (core_frame_done) begin
                    // Difference uses the first upstream acceptance, not the
                    // last sample, and includes downstream output stalls.
                    bin_latency_cycles <= cycle_counter - bin_start_cycle + 1'b1;
                    bins_completed <= bins_completed + 1'b1;
                    feature_index <= 0;
                    bin_in_flight <= 0;
                    state <= B_EMPTY;
                end
                default: state <= B_EMPTY;
            endcase
        end
    end
endmodule
