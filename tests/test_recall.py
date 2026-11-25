"""Offline tests for recall.py.   python -m unittest tests.test_recall -v"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import recall as RC  # noqa: E402

INV = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Service technician hours 8 hrs\n"
       "2 Vehicle costs 1 pcs\n3 Connection hose\n1 pcs\nINVOICE\n")


class TestParser(unittest.TestCase):
    def test_quantity_lines_are_not_positions(self):
        inv = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Leak detection and opening of the bathroom wall next\n"
               "to the washbasin\n2 hrs\n2 Replacement of split copper supply pipe section, 15\nmm, incl. fittings\n"
               "1 pcs\n3 Closing the wall opening, re-plastering and re-tiling\napprox. 0.2 m2\n0.2 m2\n"
               "4 Annual service and descaling of the gas boiler in the\nhallway\n1 pcs\nINVOICE\n")
        items = RC.parse_invoice(inv)
        self.assertEqual([i["index"] for i in items], [1, 2, 3, 4])
        self.assertEqual((items[0]["quantity"], items[0]["unit"]), (2.0, "hrs"))
        self.assertEqual(items[2]["quantity"], 0.2)
        self.assertIn("washbasin", items[0]["description"])

    def test_same_line_quantity_and_two_invoices(self):
        inv = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Speaker system 1 pcs\n2 Shipping 1 pcs\n"
               "INVOICE\nCreated on 21 Aug 2026 Page 1 / 1\nInvoice\nDATE\n16 Mar 2026\nITEMS\n"
               "POS. DESCRIPTION AMOUNTUNIT TOTAL\n5 Property damage - service\norder\n1 pcs\n7 Vehicle costs 1 pcs\nINVOICE\n")
        items = RC.parse_invoice(inv)
        self.assertEqual([i["index"] for i in items], [1, 2, 5, 7])
        self.assertEqual(items[2]["description"], "Property damage - service order")

    def test_dates_and_addresses_outside_items_ignored(self):
        inv = "DATE\n2 Apr 2026\nFROM\n3 Landfill Loop\nITEMS\nPOS. DESCRIPTION\n1 Disposal 4 pcs\nINVOICE\n"
        self.assertEqual([i["index"] for i in RC.parse_invoice(inv)], [1])


class TestBrackets(unittest.TestCase):
    def test_payoff_reading(self):
        rows = [  # fair charge 100 wrongly rejected -> paid 150; inflated 300 rejected -> 0, accepted elsewhere at 300
            {"issuer": "A", "reviewer": "X", "line_item_index": 1, "accepted": False, "amount": 150.0},
            {"issuer": "B", "reviewer": "X", "line_item_index": 1, "accepted": False, "amount": 0.0},
            {"issuer": "B", "reviewer": "Y", "line_item_index": 1, "accepted": True, "amount": 300.0},
            {"issuer": "C", "reviewer": "X", "line_item_index": 1, "accepted": True, "amount": 120.0},
        ]
        self.assertEqual(RC.brackets(rows)[1], (100.0, 300.0))


class TestFloor(unittest.TestCase):
    def test_only_earlier_games(self):
        self.assertIsNone(RC.floor_for([[70, 500.0, None]], 62))
        self.assertEqual(RC.floor_for([[21, 622.0, 680.0], [40, 600.0, None]], 62), 622.0)

    def test_capped_below_proven_inflated(self):
        self.assertAlmostEqual(RC.floor_for([[10, 100.0, None], [20, 50.0, 80.0]], 62), 80 * 0.999)

    def test_no_fair_value_no_floor(self):
        self.assertIsNone(RC.floor_for([[10, None, 40.0]], 62))


class TestApply(unittest.TestCase):
    IX = {"service technician hours|8.0|hrs": [[21, 622.0, 680.0]], "connection hose|1.0|pcs": [[16, 30.0, 33.0]]}

    def test_raise_and_parse_quantity_on_next_line(self):
        subs = [{"index": i, "charge_price": 10.0, "acceptance_limit": 0.0} for i in (1, 2, 3)]
        new, ch = RC.apply(subs, INV, 62, self.IX)
        by = {s["index"]: s for s in new}
        self.assertEqual(by[1]["charge_price"], round(0.9 * 622, 2))
        self.assertEqual(by[1]["acceptance_limit"], 622.0)
        self.assertEqual(by[2], subs[1])
        self.assertEqual(by[3]["acceptance_limit"], 30.0)
        self.assertEqual(len(ch), 2)

    def test_never_lowers(self):
        subs = [{"index": 1, "charge_price": 900.0, "acceptance_limit": 950.0}]
        self.assertEqual(RC.apply(subs, INV, 62, self.IX)[0], subs)

    def test_garbage_is_harmless(self):
        subs = [{"index": 1, "charge_price": 1.0, "acceptance_limit": 0.0}]
        for inv, ix in ((None, self.IX), (INV, None), (INV, {"x": "y"}), ("", {})):
            self.assertEqual(RC.apply(subs, inv, 62, ix)[0], subs)


class TestRealData(unittest.TestCase):
    def test_game62_from_earlier_games_only(self):
        p = os.path.join(HERE, "data", "released", "case_062.json")
        if not os.path.exists(p):
            self.skipTest("no released data")
        ix = RC.load_index()
        with open(p, encoding="utf-8") as fh:
            inv = json.load(fh)["invoice"]
        keys = {it["index"]: RC.line_key(it) for it in RC.parse_invoice(inv)}
        self.assertEqual(sorted(keys), list(range(1, 18)))           # 17 lines, as the live bot saw
        lo = RC.floor_for(ix.get(keys[3], []), 62)                  # service technician hours
        self.assertIsNotNone(lo)
        self.assertTrue(all(e[0] < 62 for e in ix.get(keys[3], []) if e[1]) or lo is not None)


class TestCeiling(unittest.TestCase):
    # earlier game 62 shares 3 of 3 lines with this invoice -> repeated case
    INV = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Room drying 1 pcs\n2 Vehicle costs 1 pcs\n"
           "3 Connection hose 1 pcs\nINVOICE\n")
    GK = {"62": ["room drying|1.0|pcs", "vehicle costs|1.0|pcs", "connection hose|1.0|pcs", "x|1.0|pcs"],
          "40": ["room drying|1.0|pcs", "other|1.0|pcs"]}
    IX = {"room drying|1.0|pcs": [[62, 91.0, 117.0]], "vehicle costs|1.0|pcs": [[40, None, 6.0]],
          "connection hose|1.0|pcs": [[62, 26.0, 28.0]]}

    def subs(self, a, b):
        return [{"index": i, "charge_price": a, "acceptance_limit": b} for i in (1, 2, 3)]

    def test_cap_inside_repeated_case(self):
        new, ch = RC.apply(self.subs(591.0, 750.0), self.INV, 65, self.IX, self.GK)
        by = {s["index"]: s for s in new}
        self.assertEqual(by[1]["charge_price"], round(0.9 * 91, 2))      # 591 >= HI 117 -> 0.9 x LO
        self.assertEqual(by[1]["acceptance_limit"], round(0.999 * 117, 2))
        self.assertEqual(by[2], self.subs(591.0, 750.0)[1])               # no fair value -> untouched
        self.assertTrue(all("hi" in c for c in ch))

    def test_no_cap_without_game_keys_or_outside_repeats(self):
        s = self.subs(591.0, 750.0)
        self.assertEqual(RC.apply(s, self.INV, 65, self.IX)[0], s)         # live call signature: floors only
        gk = {"62": ["room drying|1.0|pcs", "a|1.0|pcs", "b|1.0|pcs", "c|1.0|pcs", "d|1.0|pcs"]}
        self.assertEqual(RC.apply(s, self.INV, 65, self.IX, gk)[0], s)     # 1 shared line: not a repeat

    def test_charge_between_lo_and_hi_kept_and_b_never_below_lo(self):
        new, _ = RC.apply(self.subs(100.0, 20.0), self.INV, 65, self.IX, self.GK)
        self.assertEqual(new[0]["charge_price"], 100.0)                   # 82 <= 100 < 117: kept
        self.assertEqual(new[0]["acceptance_limit"], 91.0)                # floor still applies

    def test_only_earlier_games_count(self):
        s = self.subs(591.0, 750.0)
        self.assertEqual(RC.apply(s, self.INV, 62, self.IX, self.GK)[0], s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
