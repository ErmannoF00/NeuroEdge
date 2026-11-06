`timescale 1ns/1ps
module tb_offset_commit;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0;
    reg stage_valid=0;
    wire stage_ready;
    reg [8:0] stage_channel=0;
    reg signed [15:0] stage_value=0;
    reg abort_stage=0,update_request=0,decoder_idle=1;
    reg read_enable=0;
    reg [8:0] read_channel=0;
    wire signed [15:0] read_data;
    wire pending,write_accepted;
    wire [31:0] accepted_writes,rejected_requests,stages_accepted,aborted_stages;

    neuroedge_offset_bank dut (
        .clk(clk),.rst_n(rst_n),.stage_valid(stage_valid),.stage_ready(stage_ready),
        .stage_channel(stage_channel),.stage_value(stage_value),
        .abort_stage(abort_stage),.update_request(update_request),
        .decoder_idle(decoder_idle),.read_enable(read_enable),
        .read_channel(read_channel),.read_data(read_data),.pending(pending),
        .write_accepted(write_accepted),.accepted_writes(accepted_writes),
        .rejected_requests(rejected_requests),.stages_accepted(stages_accepted),
        .aborted_stages(aborted_stages));

    task automatic cycle(
        input reg s, input reg [8:0] ch, input reg signed [15:0] val,
        input reg abort_in,input reg req_in,input reg idle_in,
        input reg rd,input reg [8:0] rdch
    );
    begin
        @(negedge clk);
        stage_valid=s;stage_channel=ch;stage_value=val;
        abort_stage=abort_in;update_request=req_in;decoder_idle=idle_in;
        read_enable=rd;read_channel=rdch;
        @(posedge clk);#1;
    end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n=1;
        cycle(0,0,0,0,0,1,1,7);
        if(read_data!==0 || pending!==0) $fatal(1,"reset/default read");
        cycle(1,7,-30,0,0,1,0,0);
        if(!pending || stages_accepted!==1) $fatal(1,"stage failed");
        cycle(0,0,0,0,0,1,1,7);
        if(read_data!==0) $fatal(1,"staged write leaked");
        cycle(1,9,50,0,1,0,0,0);
        if(rejected_requests!==1 || stages_accepted!==1 || !pending)
            $fatal(1,"idle/write gate failed");
        cycle(0,0,0,0,1,1,1,7);
        if(!write_accepted || accepted_writes!==1 || pending || read_data!==0)
            $fatal(1,"commit/read-old semantics failed");
        cycle(0,0,0,0,0,1,1,7);
        if(read_data!==-30 || write_accepted) $fatal(1,"committed read failed");
        cycle(1,7,55,0,0,1,0,0);
        cycle(1,9,66,0,0,1,0,0);
        if(stages_accepted!==2 || !pending) $fatal(1,"pending stage overwrite");
        cycle(0,0,0,1,1,1,0,0);
        if(pending || aborted_stages!==1 || accepted_writes!==1)
            $fatal(1,"abort priority failed");
        cycle(0,0,0,0,1,1,0,0);
        if(rejected_requests!==2 || write_accepted) $fatal(1,"empty request gate failed");
        cycle(1,0,123,0,0,1,0,0);
        cycle(0,0,0,0,1,1,0,0);
        cycle(0,0,0,0,0,1,1,0);
        if(read_data!==123 || accepted_writes!==2 || stages_accepted!==3)
            $fatal(1,"second commit failed");
        @(negedge clk); rst_n=0;
        repeat (2) @(negedge clk);
        rst_n=1;
        cycle(0,0,0,0,0,1,1,7);
        if(read_data!==0 || accepted_writes!==0 || pending!==0)
            $fatal(1,"reset must clear valid bits");
        $display("OFFSET_COMMIT_RTL_PASS staged=3 committed=2 abort=1 denied=2 reset=PASS");
        $finish;
    end
endmodule
