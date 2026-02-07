-- ============================================================
-- Smart Logistics & Supply Chain Analysis
-- SQL Questions & Queries
-- Dataset table: logistics_data
-- ============================================================

/*
1) What is the overall logistics delay rate across all trips?
   - Returns total number of trips, delayed trips, and delay rate (%).
*/
SELECT
    COUNT(*) AS total_trips,
    SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) AS delayed_trips,
    ROUND(
        100.0 * SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS delay_rate_pct
FROM logistics_data;


---------------------------------------------------------------
/*
2) Which trucks (Asset_ID) have the highest delay rates?
   - Returns delay rate (%) per truck, sorted from worst to best.
*/
SELECT
    Asset_ID,
    COUNT(*) AS total_trips,
    SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) AS delayed_trips,
    ROUND(
        100.0 * SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS delay_rate_pct
FROM logistics_data
GROUP BY Asset_ID
ORDER BY delay_rate_pct DESC;


---------------------------------------------------------------
/*
3) How do different traffic conditions impact the delay rate?
   - Compares delay rate (%) for each Traffic_Status.
*/
SELECT
    Traffic_Status,
    COUNT(*) AS total_trips,
    SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) AS delayed_trips,
    ROUND(
        100.0 * SUM(CASE WHEN Logistics_Delay = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS delay_rate_pct
FROM logistics_data
GROUP BY Traffic_Status
ORDER BY delay_rate_pct DESC;


---------------------------------------------------------------
/*
4) For each truck, how many mechanical failures occur and
   what is its average asset utilization?
   - Correlates mechanical failures with avg utilization per Asset_ID.
*/
SELECT
    Asset_ID,
    COUNT(*) AS total_records,
    SUM(CASE WHEN Logistics_Delay_Reason = 'Mechanical Failure' THEN 1 ELSE 0 END)
        AS mechanical_failures,
    ROUND(AVG(Asset_Utilization), 2) AS avg_utilization_pct
FROM logistics_data
GROUP BY Asset_ID
ORDER BY mechanical_failures DESC, avg_utilization_pct DESC;


---------------------------------------------------------------
/*
5) How does each truck's average waiting time compare to the fleet average?
   - Returns avg waiting time per truck, global average and the difference.
*/
WITH global_wait AS (
    SELECT AVG(Waiting_Time) AS avg_wait_global
    FROM logistics_data
)
SELECT
    l.Asset_ID,
    AVG(l.Waiting_Time) AS avg_wait_truck,
    g.avg_wait_global,
    AVG(l.Waiting_Time) - g.avg_wait_global AS diff_from_global
FROM logistics_data l
CROSS JOIN global_wait g
GROUP BY l.Asset_ID, g.avg_wait_global
ORDER BY avg_wait_truck DESC;
