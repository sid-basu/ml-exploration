# Datetime & Time Series Cheat Sheet

---

# PANDAS / PYTHON

## Creating & Casting

```python
from datetime import datetime, timedelta
import pandas as pd

ts = datetime(2024, 3, 1, 10, 30, 0)        # create a timestamp
delta = timedelta(hours=2, minutes=30)        # create a duration
pd_delta = pd.Timedelta(minutes=30)           # pandas version

df['ts'] = pd.to_datetime(df['ts_string'])    # cast column to datetime
```

## Extracting Date Parts

```python
df['date']   = df['ts'].dt.date          # datetime.date object
df['year']   = df['ts'].dt.year          # int
df['month']  = df['ts'].dt.month         # int (1-12)
df['day']    = df['ts'].dt.day           # int (1-31)
df['hour']   = df['ts'].dt.hour          # int (0-23)
df['dow']    = df['ts'].dt.dayofweek     # int (0=Mon, 6=Sun)
df['week']   = df['ts'].dt.isocalendar().week
df['month_start'] = df['ts'].dt.to_period('M').dt.to_timestamp()  # floor to month
```

## Time Differences

```python
# Subtract two columns → timedelta Series
df['gap'] = df['end'] - df['start']

# Convert to numeric
df['gap_days']    = df['gap'].dt.days                        # whole days (floors)
df['gap_hours']   = df['gap'].dt.total_seconds() / 3600
df['gap_minutes'] = df['gap'].dt.total_seconds() / 60
df['gap_minutes'] = df['gap'] / pd.Timedelta(minutes=1)     # equivalent

# Compare to threshold
df['too_long'] = df['gap'] > pd.Timedelta(minutes=30)
```

## Notes from my experience

```python
df = df.sort_values(['user_id', 'timestamp']).copy()
df['login_date_gap'].isna()
df['row_num'] = df.groupby('user_id').cumcount() # needs to be sorted first
```

**Always sort before:** rolling, shift, diff, cumsum.
**Always `.copy()`** inside functions — avoids mutating the original DataFrame.

## groupby + transform vs agg

```python
# transform — same-length output, aligned to original index
# use when you need a new column alongside original rows
df['user_avg'] = df.groupby('user_id')['value'].transform('mean')

# agg — one row per group
# use when you want a summary table
summary = df.groupby('user_id')['value'].agg(['mean', 'sum', 'count'])

# transform with lambda (for chained operations like shift+rolling)
df['rolling_avg'] = (
    df.groupby('user_id')['value']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
```

## window function
```
 df['max_user_date'] = df.groupby('user_id')['login_date'].transform('max')
 ```

## Shift (Lag / Lead)

```python
df['prev_val']  = df.groupby('user_id')['value'].shift(1)   # lag
df['next_val']  = df.groupby('user_id')['value'].shift(-1)  # lead
df['gap']       = df.groupby('user_id')['timestamp'].diff()  # same as col - col.shift(1)
df['prev_time'] = df.groupby('user_id')['timestamp'].shift(1)
```

## Rolling Windows

```python
# Simple rolling mean (within group)
df['rm5'] = df.groupby('uid')['val'].transform(lambda x: x.rolling(5).mean())

# min_periods=1: use whatever rows are available; default requires full window
df['rm5'] = df.groupby('uid')['val'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# EXCLUDE current row from its own window — shift first:
df['prior_avg'] = (
    df.groupby('uid')['val']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Weighted rolling (e.g. weights [1,2,3] for last 3 periods)
df['weighted'] = (
    df.groupby('uid')['val']
    .transform(lambda x: x.rolling(3, min_periods=1)
               .apply(lambda w: np.dot(w, [1,2,3][-len(w):]) / sum([1,2,3][-len(w):]), raw=True))
)
```

## Cumulative Operations

```python
df['cumsum']   = df.groupby('user_id')['value'].cumsum()
df['cummax']   = df.groupby('user_id')['value'].cummax()
df['cumcount'] = df.groupby('user_id').cumcount()   # 0-indexed row number per group
df['rank']     = df.groupby('user_id')['value'].rank()
```

## Gap-Based Sessionization (MEMORIZE THIS)

```python
df = df.sort_values(['user_id', 'timestamp']).copy()

df['gap']       = df.groupby('user_id')['timestamp'].diff()
df['new_session'] = df['gap'].isna() | (df['gap'] > pd.Timedelta(minutes=30))
df['session_id']  = df['new_session'].cumsum()   # ← the key move

sessions = df.groupby(['user_id', 'session_id']).agg(
    start_time  = ('timestamp', 'min'),
    end_time    = ('timestamp', 'max'),
    event_count = ('timestamp', 'count'),
).reset_index()

sessions['duration_min'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds() / 60
```

## Streak Detection

```python
df = df.sort_values(['user_id', 'date']).copy()

# Flag rows that continue a streak (consecutive day)
df['day_diff'] = df.groupby('user_id')['date'].diff().dt.days
df['new_streak'] = (df['day_diff'] != 1) | df['day_diff'].isna()
df['streak_id']  = df.groupby('user_id')['new_streak'].cumsum()

streak_lengths = df.groupby(['user_id','streak_id']).size().reset_index(name='streak_length')
max_streaks = streak_lengths.groupby('user_id')['streak_length'].max()
```

## Week-over-Week / Lag Comparison

```python
df = df.sort_values(['group', 'date']).copy()

df['prior'] = df.groupby('group')['metric'].shift(1)
df['pct_change'] = (df['metric'] - df['prior']) / df['prior']
df['is_spike'] = df['pct_change'] > 0.20
```

## Running Total + First Threshold Crossing

```python
df['cumulative'] = df.groupby(['user_id', 'month'])['amount'].cumsum()
df['pct_of_budget'] = df['cumulative'] / df['budget']

# First date exceeding threshold per group
breaches = (
    df[df['pct_of_budget'] >= 0.80]
    .groupby(['user_id', 'month'])
    .first()
    .reset_index()
)
```

## Common Pitfalls

| Mistake | Fix |
|---|---|
| Forgetting `.copy()` | Always `df = df.copy()` inside functions |
| Rolling includes current row | `.shift(1)` before `.rolling()` |
| Not sorting before shift/rolling | `sort_values(['user_id','timestamp'])` first |
| `timedelta(a, b)` to diff timestamps | Just subtract: `a - b` |
| `.dt.days` vs `.dt.total_seconds()` | `.dt.days` = whole days; `.dt.total_seconds()` = exact float |
| `.iloc[0]` forgotten on scalar extract | `df.query(...).iloc[0]['col']` not `df.query(...)['col']` |
| `min_periods` default | Default = window size → NaN until full. Use `min_periods=1` to start early |
| Merge column collisions | Select only needed columns before merge to avoid `_x`/`_y` suffixes |
| HAVING vs WHERE | WHERE = row filter before GROUP BY; HAVING = filter after GROUP BY |

---

# DUCKDB SQL

## Date Casting & Truncation

```sql
CAST(ts AS DATE)                      -- timestamp → date
DATE_TRUNC('day', ts)                 -- floor to day (returns timestamp)
DATE_TRUNC('hour', ts)                -- floor to hour
DATE_TRUNC('week', ts)                -- floor to Monday of that week
DATE_TRUNC('month', ts)               -- floor to first of month
```

## Extracting Parts

```sql
EXTRACT(HOUR FROM ts)                 -- hour as int
EXTRACT(DOW FROM ts)                  -- day of week (0=Sun in DuckDB)
DATE_PART('month', ts)                -- same as EXTRACT
```

## Date Arithmetic

```sql
ts + INTERVAL '30 minutes'            -- add time
ts + INTERVAL '7 days'
DATE_DIFF('day', ts1, ts2)            -- integer day difference
DATE_DIFF('minute', ts1, ts2)         -- integer minute difference
DATE_DIFF('month', date1, date2)      -- integer month difference
```

## Window Functions

```sql
-- Running total
SUM(amount) OVER (PARTITION BY user_id ORDER BY date ROWS UNBOUNDED PRECEDING)

-- Rolling 7-day average
AVG(metric) OVER (PARTITION BY user_id ORDER BY date ROWS 6 PRECEDING)

-- Lag (prior row)
LAG(metric, 1) OVER (PARTITION BY user_id ORDER BY date)

-- Lead (next row)
LEAD(metric, 1) OVER (PARTITION BY user_id ORDER BY date)

-- Row number per group
ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC)

-- Proportional attribution (no subquery needed)
amount / SUM(amount) OVER (PARTITION BY user_id, date) AS proportion
```

## Top-N per Group

```sql
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY metric DESC) AS rn
    FROM my_table
)
SELECT * FROM ranked WHERE rn <= 3
```

## QUALIFY (filter on window function without CTE)

```sql
SELECT * FROM page_views
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC) = 1
```

## NULL Handling

```sql
NULLIF(denominator, 0)               -- returns NULL if 0, avoids division by zero
COALESCE(val, 0)                     -- replace NULL with 0
```

## DuckDB + Pandas

```python
con = duckdb.connect()
con.register('my_table', df)         # register DataFrame as a table
result = con.execute("SELECT ...").df()  # returns pandas DataFrame
```
