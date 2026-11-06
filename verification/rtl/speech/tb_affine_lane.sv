`timescale 1ns/1ps
module tb_affine_lane;
    reg clk=1'b0;
    always #5 clk=~clk;
    reg rst_n=1'b0;
    reg in_valid=1'b0;
    wire in_ready;
    reg signed [7:0] in_sample=0;
    reg signed [23:0] in_gain_q=0;
    reg signed [23:0] in_offset_q=0;
    wire out_valid;
    reg out_ready=1'b1;
    wire signed [7:0] out_sample;
    integer fd,count,n,scan,xi,gi,bi,expected;
    reg [1023:0] file_name;

    neuroedge_affine_lane dut(
      .clk(clk),.rst_n(rst_n),
      .in_valid(in_valid),.in_ready(in_ready),
      .in_sample(in_sample),.in_gain_q(in_gain_q),.in_offset_q(in_offset_q),
      .out_valid(out_valid),.out_ready(out_ready),.out_sample(out_sample));

    initial begin
        if (!$value$plusargs("VECTORS=%s",file_name)) $fatal(1,"Missing +VECTORS=...");
        fd=$fopen(file_name,"r");
        if (fd==0) $fatal(1,"Cannot read vectors file");
        scan=$fscanf(fd,"%d\n",count);
        if (scan!=1 || count<=0) $fatal(1,"Invalid vector count");
        repeat (3) @(negedge clk);
        rst_n=1'b1;
        for (n=0;n<count;n=n+1) begin
            scan=$fscanf(fd,"%d %d %d %d\n",xi,gi,bi,expected);
            if(scan!=4) $fatal(1,"Bad vector row %0d",n);
            @(negedge clk);
            if(!in_ready) $fatal(1,"Unexpected stall at row %0d",n);
            in_sample=xi;
            in_gain_q=gi;
            in_offset_q=bi;
            in_valid=1'b1;
            @(posedge clk);
            #1;
            if(!out_valid || $signed(out_sample) !== expected)
                $fatal(1,"AFFINE_MISMATCH row=%0d actual=%0d expected=%0d", n, $signed(out_sample),expected);
        end
        @(negedge clk);
        in_valid=1'b0;
        @(posedge clk); #1;
        if(out_valid) $fatal(1,"Output valid did not clear");
        $display("REAL_NEURAL_AFFINE_RTL_PASS samples=%0d",count);
        $fclose(fd);
        $finish;
    end
endmodule
