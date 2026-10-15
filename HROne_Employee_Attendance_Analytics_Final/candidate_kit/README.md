# Employee Attendance & Analytics API

FastAPI and MongoDB backend service implementing the complete HROne Employee Attendance & Analytics API contract (`openapi.yaml`) and business rules (R1–R10).

## Requirements

- Python 3.11+
- MongoDB 6.0+ (Tested against MongoDB 8.2)

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MONGO_URI` | MongoDB connection URI | `mongodb://localhost:27017` |
| `MONGO_DB` | MongoDB database name | `attendance_db` |

Real environment variables take precedence over `.env` configuration (`load_dotenv(override=False)`).

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Seed sample data** (Optional):
   ```bash
   python sample_seed.py
   ```

3. **Start the application**:
   ```bash
   uvicorn app.main:app --port 8000
   ```
   Or using Python module execution:
   ```bash
   python -m uvicorn app.main:app --port 8000
   ```

All database indexes (`employees`, `attendance_logs`) are created idempotently at application startup during the FastAPI `lifespan` event.

## Endpoints Implemented

- **System**:
  - `GET /health` (Readiness check: verifies MongoDB ping, returns 503 if unreachable)
- **Employees**:
  - `POST /employees` (Create employee with unique `emp_code`, validation, server-side `created_at`)
  - `GET /employees` (Paginated list, filtered by department, sorted by `emp_code ASC`)
- **Attendance**:
  - `POST /attendance/punch-in` (Race-safe punch-in with R1 date attribution and R2 late calculation)
  - `POST /attendance/punch-out` (Atomic punch-out with R3 overtime, R4 work hours, R5 half-day)
  - `GET /attendance` (Indexed pagination, date range filtering, sorted by `date DESC, emp_code ASC`)
  - `PATCH /attendance/{emp_code}/{date}` (Optimistic concurrency regularization with audit `history`)
- **Analytics (MongoDB Aggregations)**:
  - `GET /analytics/employees/{emp_code}/monthly` (Monthly summary, R7 working days, attendance rate)
  - `GET /analytics/departments/summary` (Cross-collection summary, 0-log headcount R9, record-level work hours)
  - `GET /analytics/leaderboard/late` (Standard competition ranking 1, 2, 2, 4 via `$setWindowFields`)
  - `GET /analytics/departments/{department}/trend` (Daily series gap-filling via `$densify`, 7-day trailing moving average)
- **Admin**:
  - `GET /admin/explain/{endpoint}` (`executionStats` query explain plan verifying `IXSCAN` and no `COLLSCAN`)

## Automated Tests

Run the comprehensive test suites:
```bash
python test_suite.py
python test_edge_cases.py
```

Tested test coverage includes:
- Startup and MongoDB health checks.
- Employee uniqueness, regex validations, and shift constraints.
- Grace period boundaries (09:40:00 -> 0 min; 09:40:01 -> 10 min; 10:15:59 -> 45 min).
- Overnight shifts across midnight (R1, R3).
- Concurrent race safety for punch-in (10 parallel requests -> exactly one 201, 9 return 409).
- Concurrent race safety for punch-out (10 parallel requests -> exactly one 200, 9 return 409).
- Regularization optimistic concurrency (conflicting edits return 409).
- Leaderboard competition ties (1, 2, 2, 4) at limit boundaries.
- Gap filling across weekends and zero-activity calendar days.
- Explain output index verification (`IXSCAN`).

## Known Limitations

- Audit `history` in `PATCH` relies on optimistic concurrency checking history length. Under high concurrency on the same employee-date document, conflicting writers receive HTTP 409 and must retry.
- In-memory gap filling is strictly avoided; the trend endpoint requires MongoDB 5.1+ support for the `$densify` aggregation stage.
