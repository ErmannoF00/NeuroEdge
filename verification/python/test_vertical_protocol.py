from neuroedge.vertical.protocol_model import SerialClassifierProtocol


def test_protocol_rejects_new_features_until_result_consumed():
    dut=SerialClassifierProtocol((1,-2,3,-4,5,-6,7,-8),bias=-12)
    inp=(-128, 127, 2, -3, 4, -5, 6, -7)
    for x in inp:
        assert dut.input_ready
        dut.clock(True,x)
    expected=sum(x*w for x,w in zip(inp,dut.weights))+dut.bias
    assert dut.response_valid and dut.response_score==expected
    for _ in range(8):
        dut.clock(in_valid=True,feature=15,out_ready=False)
        assert dut.response_valid and dut.response_score==expected
        assert dut.index == 0
    dut.clock(out_ready=True)
    assert not dut.response_valid and dut.input_ready


def test_protocol_reset_discards_partial_vector():
    dut=SerialClassifierProtocol(tuple(range(8)),bias=8)
    for _ in range(5):
        dut.clock(in_valid=True,feature=127)
    assert dut.index==5
    dut.clock(reset_n=False)
    assert dut.index==0 and dut.accumulated==0 and not dut.response_valid
