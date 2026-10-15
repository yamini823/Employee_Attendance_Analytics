"""
Comprehensive test suite for Employee Attendance & Analytics API
Runs all test cases against local MongoDB instance.
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

from app.main import app, client, db, IST

tc = TestClient(app)


class TestAttendanceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Trigger lifespan startup (create indexes)
        cls.client_ctx = TestClient(app)
        cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    # ----------------------------------------------------------------------- #
    # 1. Health Probe
    # ----------------------------------------------------------------------- #
    def test_01_health_ok(self):
        resp = tc.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    # ----------------------------------------------------------------------- #
    # 2. Employees Endpoints
    # ----------------------------------------------------------------------- #
    def test_02_create_employee_valid(self):
        body = {
            "emp_code": "EMP9999",
            "name": "Test User",
            "email": "test.user@example.com",
            "department": "QA",
            "shift_start": "10:00",
            "shift_end": "19:00",
            "joined_on": "2026-05-01",
        }
        resp = tc.post("/employees", json=body)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP9999")
        self.assertIsInstance(data["created_at"], int)
        self.assertGreaterEqual(data["created_at"], 100_000_000_000)

    def test_03_create_employee_duplicate(self):
        body = {
            "emp_code": "EMP0001",
            "name": "Duplicate User",
            "email": "dup@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-01",
        }
        resp = tc.post("/employees", json=body)
        self.assertEqual(resp.status_code, 409)

    def test_04_create_employee_invalid_inputs(self):
        # Invalid emp_code regex
        resp = tc.post("/employees", json={
            "emp_code": "INVALID",
            "name": "Foo",
            "email": "foo@example.com",
            "department": "QA",
            "joined_on": "2026-01-01"
        })
        self.assertEqual(resp.status_code, 422)

        # Equal shift_start and shift_end
        resp = tc.post("/employees", json={
            "emp_code": "EMP9998",
            "name": "Foo",
            "email": "foo@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "09:30",
            "joined_on": "2026-01-01"
        })
        self.assertEqual(resp.status_code, 422)

    def test_05_list_employees_pagination_and_sorting(self):
        resp = tc.get("/employees?page=1&page_size=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["page_size"], 2)
        self.assertGreaterEqual(data["total"], 6)
        self.assertEqual(len(data["items"]), 2)
        # Check ascending sort
        codes = [item["emp_code"] for item in data["items"]]
        self.assertEqual(codes, sorted(codes))

    def test_06_list_employees_filter_department(self):
        resp = tc.get("/employees?department=Engineering")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)
        for item in data["items"]:
            self.assertEqual(item["department"], "Engineering")

    # ----------------------------------------------------------------------- #
    # 3. Punch-in Endpoints & Business Rules
    # ----------------------------------------------------------------------- #
    def test_07_punch_in_unknown_employee(self):
        resp = tc.post("/attendance/punch-in", json={"emp_code": "EMP0404"})
        self.assertEqual(resp.status_code, 404)

    def test_08_punch_in_grace_period_boundaries(self):
        # EMP0001 shift is 09:30 - 18:30 IST.
        # Test A: 09:40:00 IST on 2026-07-20 (exact 10 min grace -> late_minutes = 0)
        # 2026-07-20 09:40:00 IST = 2026-07-20 04:10:00 UTC
        dt_a = datetime(2026, 7, 20, 4, 10, 0, tzinfo=timezone.utc)
        ms_a = int(dt_a.timestamp() * 1000)
        resp_a = tc.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms_a})
        self.assertEqual(resp_a.status_code, 201)
        self.assertEqual(resp_a.json()["late_minutes"], 0)
        self.assertEqual(resp_a.json()["date"], "2026-07-20")

        # Test B: 09:40:01 IST on 2026-07-21 (strictly > 10 min -> late_minutes = 10)
        # 2026-07-21 09:40:01 IST = 2026-07-21 04:10:01 UTC
        dt_b = datetime(2026, 7, 21, 4, 10, 1, tzinfo=timezone.utc)
        ms_b = int(dt_b.timestamp() * 1000)
        resp_b = tc.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms_b})
        self.assertEqual(resp_b.status_code, 201)
        self.assertEqual(resp_b.json()["late_minutes"], 10)

        # Test C: 10:15:59 IST on 2026-07-22 (late_minutes = 45)
        # 2026-07-22 10:15:59 IST = 2026-07-22 04:45:59 UTC
        dt_c = datetime(2026, 7, 22, 4, 45, 59, tzinfo=timezone.utc)
        ms_c = int(dt_c.timestamp() * 1000)
        resp_c = tc.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms_c})
        self.assertEqual(resp_c.status_code, 201)
        self.assertEqual(resp_c.json()["late_minutes"], 45)

    def test_09_punch_in_overnight_shift_date_attribution(self):
        # EMP0005 has overnight shift 22:00 - 06:00 IST.
        # Punch-in on 2026-07-25 at 05:30:00 IST (earlier than 06:00) belongs to 2026-07-24!
        # 2026-07-25 05:30:00 IST = 2026-07-25 00:00:00 UTC
        dt = datetime(2026, 7, 25, 0, 0, 0, tzinfo=timezone.utc)
        ms = int(dt.timestamp() * 1000)
        resp = tc.post("/attendance/punch-in", json={"emp_code": "EMP0005", "punched_at": ms})
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["date"], "2026-07-24")

    def test_10_punch_in_duplicate_conflict(self):
        # EMP0001 already punched in on 2026-07-20
        dt = datetime(2026, 7, 20, 5, 0, 0, tzinfo=timezone.utc)
        ms = int(dt.timestamp() * 1000)
        resp = tc.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms})
        self.assertEqual(resp.status_code, 409)

    # ----------------------------------------------------------------------- #
    # 4. Punch-out Endpoints & Calculations
    # ----------------------------------------------------------------------- #
    def test_11_punch_out_valid_and_overtime(self):
        # EMP0001 punched in on 2026-07-20 at 09:40:00 IST (04:10:00 UTC).
        # Punch out at 19:15:00 IST (13:45:00 UTC) -> 45 min overtime (>= 30 min threshold)
        # Duration: 09:40 to 19:15 = 9 hours 35 min = 9.5833... hours -> 9.58
        dt_out = datetime(2026, 7, 20, 13, 45, 0, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)
        resp = tc.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms_out})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["work_hours"], 9.58)
        self.assertEqual(data["overtime_minutes"], 45)
        self.assertEqual(data["half_day"], False)

    def test_12_punch_out_half_day_threshold(self):
        # EMP0001 punched in on 2026-07-21 at 09:40:01 IST (04:10:01 UTC).
        # Punch out after 4 hours 10 minutes = 4.17 hours (< 4.50 -> half_day = True)
        # 04:10:01 + 4h 10m = 08:20:01 UTC
        dt_out = datetime(2026, 7, 21, 8, 20, 1, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)
        resp = tc.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms_out})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["work_hours"], 4.17)
        self.assertEqual(data["half_day"], True)
        self.assertEqual(data["overtime_minutes"], 0)

    def test_13_punch_out_duplicate_conflict(self):
        # Attempt to punch out again for 2026-07-21
        dt_out = datetime(2026, 7, 21, 9, 0, 0, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)
        resp = tc.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms_out})
        self.assertEqual(resp.status_code, 409)

    def test_14_punch_out_invalid_timestamps(self):
        # Test A: punched_at == punch_in for EMP0001 on 2026-07-22 (punched in at 04:45:59 UTC)
        dt_equal = datetime(2026, 7, 22, 4, 45, 59, tzinfo=timezone.utc)
        ms_equal = int(dt_equal.timestamp() * 1000)
        resp_equal = tc.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms_equal})
        self.assertEqual(resp_equal.status_code, 422)

        # Test B: punched_at < punch_in for EMP0006 (punched in at 2026-07-07 04:05:00 UTC)
        # Attempt punch out at 04:00:00 UTC
        dt_early = datetime(2026, 7, 7, 4, 0, 0, tzinfo=timezone.utc)
        ms_early = int(dt_early.timestamp() * 1000)
        resp_early = tc.post("/attendance/punch-out", json={"emp_code": "EMP0006", "punched_at": ms_early})
        self.assertEqual(resp_early.status_code, 422)

        # Test C: Punched_at > 24 hours after punch_in for EMP0001
        dt_late = datetime(2026, 7, 24, 0, 0, 0, tzinfo=timezone.utc)
        ms_late = int(dt_late.timestamp() * 1000)
        resp_late = tc.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms_late})
        self.assertEqual(resp_late.status_code, 422)

    # ----------------------------------------------------------------------- #
    # 5. List Attendance & Regularization
    # ----------------------------------------------------------------------- #
    def test_15_list_attendance(self):
        resp = tc.get("/attendance?emp_code=EMP0001&page=1&page_size=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total"], 1)
        for item in data["items"]:
            self.assertEqual(item["emp_code"], "EMP0001")
            self.assertNotIn("_id", item)
            self.assertNotIn("id", item)

    def test_16_list_attendance_date_range_invalid(self):
        resp = tc.get("/attendance?date_from=2026-07-10&date_to=2026-07-01")
        self.assertEqual(resp.status_code, 422)

    def test_17_regularize_attendance_success(self):
        # Regularize EMP0001 on 2026-07-22: set punch_out
        dt_out = datetime(2026, 7, 22, 13, 0, 0, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)
        body = {
            "punch_out": ms_out,
            "reason": "Forgot to punch out biometric glitch",
            "regularized_by": "hr_manager",
        }
        resp = tc.patch("/attendance/EMP0001/2026-07-22", json=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["history"]), 1)
        self.assertEqual(data["history"][0]["by"], "hr_manager")
        self.assertIn("punch_out", data["history"][0]["changes"])
        self.assertIn("work_hours", data["history"][0]["changes"])

    def test_18_regularize_attendance_no_change(self):
        # Calling regularize with same parameters -> changes nothing -> 422
        body = {
            "reason": "Doing nothing at all here",
            "regularized_by": "hr_manager",
        }
        resp = tc.patch("/attendance/EMP0001/2026-07-22", json=body)
        self.assertEqual(resp.status_code, 422)

    # ----------------------------------------------------------------------- #
    # 6. Analytics Endpoints
    # ----------------------------------------------------------------------- #
    def test_19_employee_monthly(self):
        resp = tc.get("/analytics/employees/EMP0001/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["emp_code"], "EMP0001")
        self.assertEqual(data["month"], "2026-07")
        self.assertEqual(data["working_days"], 23)  # July 2026 has 23 weekdays
        self.assertGreaterEqual(data["present_days"], 1.0)
        self.assertIsInstance(data["attendance_pct"], float)

    def test_20_department_summary(self):
        resp = tc.get("/analytics/departments/summary?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["month"], "2026-07")
        departments = [item["department"] for item in data["items"]]
        self.assertIn("Engineering", departments)
        self.assertIn("Sales", departments)
        self.assertIn("Support", departments)
        for d in data["items"]:
            self.assertGreater(d["headcount"], 0)

    def test_21_late_leaderboard(self):
        resp = tc.get("/analytics/leaderboard/late?month=2026-07&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["month"], "2026-07")
        ranks = [item["rank"] for item in data["items"]]
        # Standard competition ranks must start at 1
        if ranks:
            self.assertEqual(ranks[0], 1)
        for item in data["items"]:
            self.assertGreater(item["total_late_minutes"], 0)

    def test_22_department_trend(self):
        resp = tc.get("/analytics/departments/Sales/trend?from=2026-07-06&to=2026-07-12")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["department"], "Sales")
        # Exact 7 calendar days
        self.assertEqual(len(data["items"]), 7)
        dates = [item["date"] for item in data["items"]]
        self.assertEqual(dates, [
            "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-12"
        ])
        for item in data["items"]:
            if not item["is_working_day"]:
                self.assertIsNone(item["attendance_rate"])

    # ----------------------------------------------------------------------- #
    # 7. Admin Explain Endpoint
    # ----------------------------------------------------------------------- #
    def test_23_admin_explain_endpoints(self):
        # 1. attendance_list
        resp = tc.get("/admin/explain/attendance_list?emp_code=EMP0001")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["collection"], "attendance_logs")
        self.assertIn("queryPlanner", resp.json()["explain"])

        # 2. employee_monthly
        resp = tc.get("/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["collection"], "attendance_logs")

        # 3. department_summary
        resp = tc.get("/admin/explain/department_summary?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["collection"], "employees")

        # 4. late_leaderboard
        resp = tc.get("/admin/explain/late_leaderboard?month=2026-07")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["collection"], "attendance_logs")

        # 5. department_trend
        resp = tc.get("/admin/explain/department_trend?department=Sales&from=2026-07-01&to=2026-07-07")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["collection"], "employees")

    # ----------------------------------------------------------------------- #
    # 8. Concurrency & Race Condition Verification
    # ----------------------------------------------------------------------- #
    def test_24_concurrent_punch_in(self):
        # Attempt 10 simultaneous punch-in requests for EMP0002 on 2026-07-28
        dt = datetime(2026, 7, 28, 4, 0, 0, tzinfo=timezone.utc)
        ms = int(dt.timestamp() * 1000)

        def make_punch_in():
            return tc.post("/attendance/punch-in", json={"emp_code": "EMP0002", "punched_at": ms})

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_punch_in) for _ in range(10)]
            results = [f.result() for f in futures]

        statuses = [r.status_code for r in results]
        # Exactly one must succeed with 201, rest 409
        self.assertEqual(statuses.count(201), 1)
        self.assertEqual(statuses.count(409), 9)

    def test_25_concurrent_punch_out(self):
        # Attempt 10 simultaneous punch-out requests for EMP0002 on 2026-07-28
        dt_out = datetime(2026, 7, 28, 13, 0, 0, tzinfo=timezone.utc)
        ms_out = int(dt_out.timestamp() * 1000)

        def make_punch_out():
            return tc.post("/attendance/punch-out", json={"emp_code": "EMP0002", "punched_at": ms_out})

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_punch_out) for _ in range(10)]
            results = [f.result() for f in futures]

        statuses = [r.status_code for r in results]
        # Exactly one must succeed with 200, rest 409
        self.assertEqual(statuses.count(200), 1)
        self.assertEqual(statuses.count(409), 9)

    # ----------------------------------------------------------------------- #
    # 9. Additional Boundary & Error Contract Validations
    # ----------------------------------------------------------------------- #
    def test_26_additional_boundary_validations(self):
        # 1. Page < 1 -> 422
        resp = tc.get("/employees?page=0")
        self.assertEqual(resp.status_code, 422)

        # 2. Page size > 100 -> 422
        resp = tc.get("/employees?page_size=101")
        self.assertEqual(resp.status_code, 422)

        # 3. Monthly analytics for unknown valid-format employee -> 404
        resp = tc.get("/analytics/employees/EMP9998/monthly?month=2026-07")
        self.assertEqual(resp.status_code, 404)

        # 4. Department trend for unknown department -> 404
        resp = tc.get("/analytics/departments/UnknownDept/trend?from=2026-07-01&to=2026-07-07")
        self.assertEqual(resp.status_code, 404)

        # 5. Department trend with inverted date range -> 422
        resp = tc.get("/analytics/departments/Sales/trend?from=2026-07-10&to=2026-07-01")
        self.assertEqual(resp.status_code, 422)

        # 6. Punch-out for unknown employee -> 404
        resp = tc.post("/attendance/punch-out", json={"emp_code": "EMP9998", "punched_at": 1783312500000})
        self.assertEqual(resp.status_code, 404)

        # 7. Regularization with punch_out <= punch_in -> 422
        resp = tc.patch("/attendance/EMP0001/2026-07-07", json={
            "punch_in": 1783312500000,
            "punch_out": 1783312400000,
            "regularized_by": "admin",
            "reason": "invalid punch out before punch in"
        })
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
