// Research policy controller. Does NOT suppress inference or modify weights.
module neuroedge_budget_controller #(
  parameter integer SCORE_W = 16,
  parameter integer WINDOW_BINS = 50,
  parameter integer MAX_WRITES = 4,
  parameter integer COOLDOWN_BINS = 5,
  parameter integer HIGH_THRESHOLD = 1200,
  parameter integer LOW_THRESHOLD = 800
)(
  input wire clk,
  input wire rst_n,
  input wire sample_valid,
  input wire [SCORE_W-1:0] drift_score,
  output reg drift_active,
  output reg update_request,
  output reg [15:0] writes_in_window,
  output reg [31:0] observed_bins,
  output reg [31:0] suppressed_updates
);
  reg [31:0] window_position;
  reg [31:0] cooldown;
  reg next_active;
  reg [15:0] next_writes;
  reg [31:0] next_cooldown;
  reg [31:0] next_window;
  always @* begin
    next_active=drift_active;
    if (drift_score >= HIGH_THRESHOLD) next_active=1'b1;
    else if (drift_score <= LOW_THRESHOLD) next_active=1'b0;

    next_window = window_position;
    next_writes = writes_in_window;
    if (window_position == WINDOW_BINS-1) begin
      next_window=0;
      next_writes=0;
    end else begin
      next_window=window_position+1;
    end
    next_cooldown=(cooldown>0) ? cooldown-1 : 0;
  end
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      drift_active<=0;
      update_request<=0;
      writes_in_window<=0;
      observed_bins<=0;
      suppressed_updates<=0;
      window_position<=0;
      cooldown<=0;
    end else begin
      update_request<=0;
      if(sample_valid) begin
        observed_bins<=observed_bins+1;
        drift_active<=next_active;
        window_position<=next_window;
        writes_in_window<=next_writes;
        cooldown<=next_cooldown;
        if(next_active) begin
          if(next_writes<MAX_WRITES && cooldown==0) begin
            update_request<=1;
            writes_in_window<=next_writes+1;
            cooldown<=COOLDOWN_BINS;
          end else begin
            suppressed_updates<=suppressed_updates+1;
          end
        end
      end
    end
  end
endmodule
