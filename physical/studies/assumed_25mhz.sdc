# existing *target* constraint only. NOT achieved clock, STA or signoff.
# 20 ms/binned neural feature stream; 25MHz => 500,000 cycles per bin.
create_clock -name clk -period 40.000 [get_ports clk]
set_input_delay 2.0 -clock clk [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_output_delay 2.0 -clock clk [all_outputs]
