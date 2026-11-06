module delta_encoder #(
  parameter int W = 16,
  parameter int signed THRESH = 32
)(
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  valid_i,
  input  logic signed [W-1:0]   sample_i,
  output logic                  valid_o,
  output logic signed [1:0]     event_o
);
  logic signed [W-1:0] prev_q;
  logic signed [W:0] diff;
  always_comb diff = $signed(sample_i) - $signed(prev_q);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin prev_q <= '0; valid_o <= 1'b0; event_o <= '0; end
    else begin
      valid_o <= valid_i;
      if (valid_i) begin
        prev_q <= sample_i;
        if (diff > THRESH) event_o <= 2'sd1;
        else if (diff < -THRESH) event_o <= -2'sd1;
        else event_o <= 2'sd0;
      end
    end
  end
endmodule
