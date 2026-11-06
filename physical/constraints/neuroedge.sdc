create_clock -name clk -period 20.000 [get_ports clk]
set_input_delay 2.0 -clock clk [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_output_delay 2.0 -clock clk [all_outputs]
