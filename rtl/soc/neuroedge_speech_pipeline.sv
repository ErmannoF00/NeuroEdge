`timescale 1ns/1ps
// existing additive integration. Proven existing core and existing bank are unmodified.
// Policy: authorization is rejected unless a pending candidate and safe edge
// already exist; no queued requests. Mid-bin staging is permitted, not commit.
// Priority at one edge: reset > abort > authorization; feature acceptance
// prevents authorization; configuration writes require a separately safe edge.
module neuroedge_speech_pipeline (
    input logic clk, rst_n,
    input logic cfg_weight_we,
    input logic [15:0] cfg_weight_addr,
    input logic signed [7:0] cfg_weight_data,
    input logic cfg_meta_we,
    input logic [8:0] cfg_meta_addr,
    input logic signed [31:0] cfg_meta_bias,
    input logic [4:0] cfg_meta_shift,
    output logic cfg_ready,
    input logic feature_valid,
    output logic feature_ready,
    input logic signed [7:0] feature_data,
    input logic stage_valid,
    output logic stage_ready,
    input logic [8:0] stage_channel,
    input logic signed [15:0] stage_value,
    input logic abort_stage,
    input logic authorize_update,
    output logic request_accepted,
    output logic request_rejected,
    output logic [31:0] accepted_requests,
    output logic [31:0] rejected_requests,
    output logic [31:0] queued_requests,
    output logic pending,
    output logic write_accepted,
    output logic [31:0] accepted_writes,
    output logic [31:0] stages_accepted,
    output logic [31:0] aborted_stages,
    output logic bin_in_flight,
    output logic [8:0] feature_index,
    output logic [31:0] bin_latency_cycles,
    output logic [31:0] bins_completed,
    output logic logit_valid,
    input logic logit_ready,
    output logic [5:0] logit_index,
    output logic signed [15:0] logit_data,
    output logic [31:0] logit_frame,
    output logic frame_done,
    output logic busy
);
    logic bridge_empty;
    logic read_enable;
    logic [8:0] read_channel;
    logic signed [15:0] read_data;
    logic core_in_valid, core_in_ready;
    logic signed [7:0] core_in_data;
    logic core_busy,core_frame_done;
    logic safe_boundary;
    logic cfg_w_en,cfg_m_en;
    logic bank_commit_req;
    logic bank_stage_ready;
    wire [31:0] bank_rejected_requests; // bank never sees an invalid request

    // cfg_ready is an edge-level acceptance qualification; host MUST present
    // configuration writes only when cfg_ready and must not assume buffering.
    assign cfg_ready = rst_n && bridge_empty && !bin_in_flight &&
                       !core_busy && !core_frame_done &&
                       !feature_valid && !authorize_update && !abort_stage;
    assign cfg_w_en = cfg_ready && cfg_weight_we;
    assign cfg_m_en = cfg_ready && cfg_meta_we;
    assign safe_boundary = rst_n && bridge_empty && !bin_in_flight &&
                           !core_busy && !core_frame_done &&
                           !read_enable && !feature_valid &&
                           !cfg_weight_we && !cfg_meta_we && !abort_stage;
    assign bank_commit_req = authorize_update && pending && safe_boundary;
    assign stage_ready = rst_n && bank_stage_ready;
    // Clocked response pulses are observable for one full cycle after the
    // accepted/rejected authorization edge, avoiding half-cycle glitches when
    // the bank consumes its pending stage on the same edge.
    // No queue is implemented. Request rejection is explicit and counted.
    assign queued_requests = 32'd0;
    assign frame_done = core_frame_done;
    assign busy = core_busy || bin_in_flight;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accepted_requests <= 0;
            rejected_requests <= 0;
            request_accepted <= 0;
            request_rejected <= 0;
        end else begin
            request_accepted <= bank_commit_req;
            request_rejected <= authorize_update && !bank_commit_req;
            if (bank_commit_req) accepted_requests <= accepted_requests + 1'b1;
            if (authorize_update && !bank_commit_req)
                rejected_requests <= rejected_requests + 1'b1;
        end
    end

    neuroedge_offset_stream_bridge u_bridge (
        .clk(clk),.rst_n(rst_n),
        .feature_valid(feature_valid),.feature_ready(feature_ready),
        .feature_data(feature_data),
        .offset_read_enable(read_enable),.offset_read_channel(read_channel),
        .offset_read_data(read_data),
        .core_in_valid(core_in_valid),.core_in_ready(core_in_ready),
        .core_in_data(core_in_data),.core_frame_done(core_frame_done),
        .allow_features(!cfg_weight_we && !cfg_meta_we && !authorize_update && !abort_stage),
        .bin_in_flight(bin_in_flight),.bridge_empty(bridge_empty),
        .feature_index(feature_index),
        .bin_latency_cycles(bin_latency_cycles),.bins_completed(bins_completed)
    );

    neuroedge_offset_bank u_offsets (
        .clk(clk),.rst_n(rst_n),.stage_valid(stage_valid && rst_n),
        .stage_ready(bank_stage_ready),.stage_channel(stage_channel),
        .stage_value(stage_value),.abort_stage(abort_stage),
        .update_request(bank_commit_req),.decoder_idle(safe_boundary),
        .read_enable(read_enable),.read_channel(read_channel),
        .read_data(read_data),.pending(pending),
        .write_accepted(write_accepted),.accepted_writes(accepted_writes),
        .rejected_requests(bank_rejected_requests),
        .stages_accepted(stages_accepted),.aborted_stages(aborted_stages)
    );
    neuroedge_speech_decoder u_decoder (
        .clk(clk),.rst_n(rst_n),
        .cfg_weight_we(cfg_w_en),.cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .cfg_meta_we(cfg_m_en),.cfg_meta_addr(cfg_meta_addr),
        .cfg_meta_bias(cfg_meta_bias),.cfg_meta_shift(cfg_meta_shift),
        .in_valid(core_in_valid),.in_ready(core_in_ready),
        .in_data(core_in_data),.logit_valid(logit_valid),
        .logit_ready(logit_ready),.logit_index(logit_index),
        .logit_data(logit_data),.logit_frame(logit_frame),
        .frame_done(core_frame_done),.busy(core_busy)
    );
endmodule
