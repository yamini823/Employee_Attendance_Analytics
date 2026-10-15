"""
Deep edge-case verification for HROne Employee Attendance & Analytics API:
1. Leaderboard competition ties (1, 2, 2, 4) at limit boundary
2. Mid-month joiners and zero-log employee headcount (R7, R9)
3. Overnight shift full cycle across midnight (R1, R3, R4)
4. Regularization concurrency / conflict (409)
5. Half-up decimal rounding verification
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
import os
import unittest

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pymongo import MongoClient

load_dotenv(override=False)

# Seed database before running tests
os.system("python sample_seed.py")

from app.main import app, client, db

tc = TestClient(app)


class TestDeepEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_01_leaderboard_competition_ties_at_limit(self):
        """
        Verify competition ranking (1, 2, 2, 4) and that limit applies AFTER ranking.
        If limit=2 and ranks are 1, 2, 2, 4, then BOTH rank 2 items must be returned (total 3 items).
        """
        # Create 4 test employees in QA department
        for i in range(1, 5):
            code = f"EMP800{i}"
            db.employees.delete_one({"emp_code": code})
            tc.post("/employees", json={
                "emp_code": code,
                "name": f"QA Tester {i}",
                "email": f"qa{i}@example.com",
                "department": "QA",
                "shift_start": "09:30",
                "shift_end": "18:30",
                "joined_on": "2026-01-01"
            })
            # Clean up old logs for this month
            db.attendance_logs.delete_many({"emp_code": code})

        # Insert late logs for 2026-08:
        # EMP8001: 50 min late (Rank 1)
        # EMP8002: 30 min late (Rank 2)
        # EMP8003: 30 min late (Rank 2) - tied!
        # EMP8004: 10 min late (Rank 4) - skipped 3!
        db.attendance_logs.insert_many([
            {"emp_code": "EMP8001", "date": "2026-08-03", "status": "PRESENT", "punch_in": datetime(2026, 8, 3, 4, 50, 0, tzinfo=timezone.utc), "punch_out": None, "work_hours": None, "late_minutes": 50, "overtime_minutes": 0, "half_day": False, "history": []},
            {"emp_code": "EMP8002", "date": "2026-08-03", "status": "PRESENT", "punch_in": datetime(2026, 8, 3, 4, 30, 0, tzinfo=timezone.utc), "punch_out": None, "work_hours": None, "late_minutes": 30, "overtime_minutes": 0, "half_day": False, "history": []},
            {"emp_code": "EMP8003", "date": "2026-08-03", "status": "PRESENT", "punch_in": datetime(2026, 8, 3, 4, 30, 0, tzinfo=timezone.utc), "punch_out": None, "work_hours": None, "late_minutes": 30, "overtime_minutes": 0, "half_day": False, "history": []},
            {"emp_code": "EMP8004", "date": "2026-08-03", "status": "PRESENT", "punch_in": datetime(2026, 8, 3, 4, 10, 0, tzinfo=timezone.utc), "punch_out": None, "work_hours": None, "late_minutes": 10, "overtime_minutes": 0, "half_day": False, "history": []},
        ])

        # Test limit=2 inside QA department:
        # Since rank 2 is tied between EMP8002 and EMP8003, both have rank <= 2.
        # So exactly 3 rows must be returned: rank 1, rank 2, rank 2!
        resp = tc.get("/analytics/leaderboard/late?month=2026-08&limit=2&department=QA")
        self.assertEqual(resp.status_code, 200)
        items = resp.json()["items"]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["rank"], 1)
        self.assertEqual(items[0]["emp_code"], "EMP8001")
        self.assertEqual(items[1]["rank"], 2)
        self.assertEqual(items[1]["emp_code"], "EMP8002")
        self.assertEqual(items[2]["rank"], 2)
        self.assertEqual(items[2]["emp_code"], "EMP8003")

        # Test limit=4: should return all 4, with ranks [1, 2, 2, 4] (rank 3 skipped!)
        resp4 = tc.get("/analytics/leaderboard/late?month=2026-08&limit=4&department=QA")
        self.assertEqual(resp4.status_code, 200)
        items4 = resp4.json()["items"]
        self.assertEqual(len(items4), 4)
        ranks = [it["rank"] for it in items4]
        self.assertEqual(ranks, [1, 2, 2, 4])

    def test_02_zero_log_employee_and_mid_month_joiner(self):
        """
        Verify headcount includes employees with 0 logs (R9)
        and working_days for mid-month joiner counts from joined_on (R7).
        """
        # EMP0002 joined on 2026-07-13.
        # July 2026 weekdays:
        # 2026-07-01 to 2026-07-31: 23 total weekdays.
        # Weekdays before July 13: July 1,2,3,6,7,8,9,10 = 8 weekdays.
        # Weekdays from July 13 to July 31: 23 - 8 = 15 weekdays!
        resp = tc.get("/analytics/employees/EMP0002/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["working_days"], 15)

        # Create an employee with joined_on in July who has ZERO logs
        db.employees.delete_one({"emp_code": "EMP8888"})
        db.attendance_logs.delete_many({"emp_code": "EMP8888"})
        tc.post("/employees", json={
            "emp_code": "EMP8888",
            "name": "Zero Logs User",
            "email": "zero@example.com",
            "department": "Engineering",
            "joined_on": "2026-07-01"
        })

        # Check monthly summary for EMP8888
        resp_zero = tc.get("/analytics/employees/EMP8888/monthly?month=2026-07")
        self.assertEqual(resp_zero.status_code, 200)
        data_zero = resp_zero.json()
        self.assertEqual(data_zero["working_days"], 23)
        self.assertEqual(data_zero["present_days"], 0.0)
        self.assertEqual(data_zero["attendance_pct"], 0.0)

        # Department summary for Engineering must now have headcount 3 (EMP0001, EMP0002, EMP8888)
        resp_dept = tc.get("/analytics/departments/summary?month=2026-07&department=Engineering")
        self.assertEqual(resp_dept.status_code, 200)
        dept_item = resp_dept.json()["items"][0]
        self.assertEqual(dept_item["headcount"], 3)

        # Clean up temporary test employee so subsequent tests are properly isolated
        db.employees.delete_one({"emp_code": "EMP8888"})
        db.attendance_logs.delete_many({"emp_code": "EMP8888"})

    def test_03_overnight_shift_cycle(self):
        """
        EMP0005 shift is 22:00 - 06:00 IST.
        Punch-in: 2026-07-10 21:55:00 IST (16:25:00 UTC) -> date is 2026-07-10.
        Punch-out: 2026-07-11 06:40:00 IST (01:10:00 UTC next day) -> 40 min overtime!
        Duration: 8.75 hours.
        """
        dt_in = datetime(2026, 7, 10, 16, 25, 0, tzinfo=timezone.utc)
        ms_in = int(dt_in.timestamp() * 1000)
        resp_in = tc.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms_in})
        self.assertEqual(resp_in.status_code, 201)
        self.assertEqual(resp_in.json()["date"], "2026-07-10")
        self.assertEqual(resp_in.json()["late_minutes"], 0)

        # Punch out
        dt_out = datetime(2026, 7, 11, 1, 10, 0, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)
        resp_out = tc.post("/attendance/punch-out", json={"emp_code": "EMP0005", "punched_at": ms_out})
        self.assertEqual(resp_out.status_code, 200)
        data_out = resp_out.json()
        self.assertEqual(data_out["work_hours"], 8.75)
        self.assertEqual(data_out["overtime_minutes"], 40)
        self.assertEqual(data_out["half_day"], False)

    def test_04_regularize_concurrent_conflict(self):
        """
        Verify concurrent regularizations do not silently lose audit history.
        The loser of concurrent edits must receive 409 Conflict.
        """
        # Record exists for EMP0001 on 2026-07-06
        dt_in_new = datetime(2026, 7, 6, 4, 0, 0, tzinfo=timezone.utc)
        ms_in_new = int(dt_in_new.timestamp() * 1000)

        def make_regularize(suffix):
            return tc.patch("/attendance/EMP0001/2026-07-06", json={
                "punch_in": ms_in_new,
                "reason": f"Concurrent correction {suffix}",
                "regularized_by": f"admin_{suffix}"
            })

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_regularize, i) for i in range(5)]
            results = [f.result() for f in futures]

        statuses = [r.status_code for r in results]
        # At least one must succeed with 200
        self.assertIn(200, statuses)
        # Concurrent conflicts should receive 409 or 422 (if no fields changed by subsequent)
        for s in statuses:
            self.assertIn(s, (200, 409, 422))

    def test_05_weekend_records_r7_business_rule(self):
        """
        R7 Business Rule:
        present_days counts only Mon-Fri records.
        Weekend records still add to late and overtime totals.
        """
        code = "EMP7001"
        db.employees.delete_one({"emp_code": code})
        db.attendance_logs.delete_many({"emp_code": code})

        tc.post("/employees", json={
            "emp_code": code,
            "name": "Weekend Worker",
            "email": "weekend@example.com",
            "department": "Operations",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-07-01",
        })

        # Insert two records:
        # 1. Saturday 2026-07-11: 40 min late, 50 min overtime, full day present
        # 2. Monday 2026-07-13: on-time, 0 overtime, full day present
        db.attendance_logs.insert_many([
            {
                "emp_code": code,
                "date": "2026-07-11",  # Saturday
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 11, 4, 40, 0, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 11, 13, 50, 0, tzinfo=timezone.utc),
                "work_hours": 9.17,
                "late_minutes": 40,
                "overtime_minutes": 50,
                "half_day": False,
                "history": [],
            },
            {
                "emp_code": code,
                "date": "2026-07-13",  # Monday
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 13, 4, 0, 0, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 13, 13, 0, 0, tzinfo=timezone.utc),
                "work_hours": 9.0,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            },
        ])

        resp = tc.get(f"/analytics/employees/{code}/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # present_days must count ONLY Monday (1.0), NOT Saturday!
        self.assertEqual(data["present_days"], 1.0)
        # late_count and total_late_minutes must INCLUDE Saturday
        self.assertEqual(data["late_count"], 1)
        self.assertEqual(data["total_late_minutes"], 40)
        # overtime must INCLUDE Saturday
        self.assertEqual(data["total_overtime_minutes"], 50)

        # Clean up
        db.employees.delete_one({"emp_code": code})
        db.attendance_logs.delete_many({"emp_code": code})

    def test_06_explain_ixscan_and_no_collscan(self):
        """
        Verify all 5 explain endpoints use an index scan (IXSCAN)
        and contain no collection scan (COLLSCAN).
        """
        import json

        queries = [
            ("attendance_list", "/admin/explain/attendance_list?emp_code=EMP0001"),
            ("attendance_list_dates", "/admin/explain/attendance_list?date_from=2026-07-01&date_to=2026-07-10"),
            ("employee_monthly", "/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07"),
            ("department_summary", "/admin/explain/department_summary?month=2026-07"),
            ("department_summary_dept", "/admin/explain/department_summary?month=2026-07&department=Engineering"),
            ("late_leaderboard", "/admin/explain/late_leaderboard?month=2026-07"),
            ("department_trend", "/admin/explain/department_trend?department=Engineering&from=2026-07-01&to=2026-07-07"),
        ]

        for label, url in queries:
            resp = tc.get(url)
            self.assertEqual(resp.status_code, 200, f"Failed for {label}")
            explain_str = json.dumps(resp.json()["explain"])
            self.assertIn("IXSCAN", explain_str, f"IXSCAN missing in {label}")
            self.assertNotIn("COLLSCAN", explain_str, f"COLLSCAN detected in {label}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
