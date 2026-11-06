// Receives 512 signed INT8 quantized neural features per 20 ms bin.
// Computes excursion count; feeds unchanged existing budget policy.
// This circuit does not skip inference or write memory. No drift claim.
module neuroedge_neural_activity_monitor #(
  parameter integer INPUTS=512,
  parameter integer MAG_THRESHOLD=32,
  parameter integer HIGH_THRESHOLD=32,
  parameter integer LOW_THRESHOLD=20,
  parameter integer WINDOW_BINS=50,
  parameter integer MAX_REQUESTS=4,
  parameter integer COOLDOWN_BINS=5
)(
  input wire clk,
  input wire rst_n,
  input wire feature_valid,
  input wire signed [7:0] feature_data,
  output reg score_valid,
  output reg [15:0] score,
  output wire drift_active,
  output wire update_request,
  output wire [15:0] requests_in_window,
  output wire [31:0] observed_bins,
  output wire [31:0] suppressed_requests
);
  localparam integer COUNT_W=$clog2(INPUTS+1);
  reg [COUNT_W-1:0] feature_pos;
  reg [COUNT_W-1:0] count;
  wire signed [8:0] extended = {feature_data[7],feature_data};
  wire [8:0] magnitude = extended[8] ? $unsigned(-extended) : $unsigned(extended);
  wire excursion=(magnitude >= MAG_THRESHOLD);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      feature_pos<=0;
      count<=0;
      score<=0;
      score_valid<=0;
    end else begin
      score_valid<=0;
      if (feature_valid) begin
        if (feature_pos==INPUTS-1) begin
          score<={{(16-COUNT_W){1'b0}},count}+excursion;
          score_valid<=1;
          count<=0;
          feature_pos<=0;
        end else begin
          feature_pos<=feature_pos+1'b1;
          if (excursion) count<=count+1'b1;
        end
      end
    end
  end

  neuroedge_budget_controller #(
    .WINDOW_BINS(WINDOW_BINS), .MAX_WRITES(MAX_REQUESTS),
    .COOLDOWN_BINS(COOLDOWN_BINS), .HIGH_THRESHOLD(HIGH_THRESHOLD),
    .LOW_THRESHOLD(LOW_THRESHOLD)
  ) ctrl (
    .clk(clk), .rst_n(rst_n), .sample_valid(score_valid),
    .drift_score(score), .drift_active(drift_active),
    .update_request(update_request), .writes_in_window(requests_in_window),
    .observed_bins(observed_bins), .suppressed_updates(suppressed_requests)
  );
endmodule
