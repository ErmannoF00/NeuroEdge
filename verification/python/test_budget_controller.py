from neuroedge.control.budget import BudgetController

def test_reset_and_hysteresis():
    c=BudgetController()
    assert c.sample(900)[0] == 0
    assert c.sample(1200)[0] == 1
    assert c.sample(900)[0] == 1
    assert c.sample(800)[0] == 0

def test_budget_and_cooldown():
    c=BudgetController(window_bins=100, max_writes=2, cooldown_bins=2)
    results=[c.sample(1500) for _ in range(30)]
    assert sum(x[1] for x in results)==2
    assert c.suppressed==28

def test_idle_is_free():
    c=BudgetController()
    for _ in range(20):
        assert c.sample(0)[1]==0
    assert c.writes==0
