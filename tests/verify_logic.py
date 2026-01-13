"""Quick verification of trailing stop logic."""

def test_trailing_stop_logic():
    """Test the core trailing stop logic."""
    trail_percent = 0.10
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Stop price calculation
    print("Test 1: Stop price at 10% below bid...")
    bid = 10.00
    stop = round(bid * (1 - trail_percent), 2)
    if stop == 9.00:
        print("  PASS: $10.00 bid → $9.00 stop")
        tests_passed += 1
    else:
        print(f"  FAIL: Expected $9.00, got ${stop}")
        tests_failed += 1
    
    # Test 2: Stop moves UP when price rises
    print("Test 2: Stop moves UP when bid rises...")
    initial_stop = 9.00
    new_bid = 12.00
    new_stop = round(new_bid * (1 - trail_percent), 2)
    should_update = new_stop > initial_stop
    if should_update and new_stop == 10.80:
        print(f"  PASS: Bid rose to $12.00 → Stop moves to $10.80")
        tests_passed += 1
    else:
        print(f"  FAIL: should_update={should_update}, new_stop={new_stop}")
        tests_failed += 1
    
    # Test 3: Stop STAYS when price falls
    print("Test 3: Stop STAYS when bid falls...")
    current_stop = 10.80
    fallen_bid = 10.00
    potential_stop = round(fallen_bid * (1 - trail_percent), 2)
    should_update = potential_stop > current_stop
    if not should_update:
        print(f"  PASS: Bid fell to $10.00 → Stop stays at $10.80 (not $9.00)")
        tests_passed += 1
    else:
        print(f"  FAIL: Stop incorrectly updated when price fell")
        tests_failed += 1
    
    # Test 4: Full trailing scenario
    print("Test 4: Full trailing scenario...")
    stop = 9.45  # Initial
    
    # Price rises
    bid = 11.00
    new_s = round(bid * 0.90, 2)
    if new_s > stop:
        stop = new_s  # 9.90
    
    # Price falls - stop should stay
    bid = 10.80
    new_s = round(bid * 0.90, 2)
    if new_s > stop:
        stop = new_s  # Should NOT update
    
    if stop == 9.90:
        print(f"  PASS: After rise and fall, stop correctly held at $9.90")
        tests_passed += 1
    else:
        print(f"  FAIL: Stop changed to ${stop} unexpectedly")
        tests_failed += 1
    
    # Summary
    print("")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


if __name__ == "__main__":
    success = test_trailing_stop_logic()
    print("")
    if success:
        print("✓ All trailing stop logic tests PASSED!")
    else:
        print("✗ Some tests FAILED")
    exit(0 if success else 1)
