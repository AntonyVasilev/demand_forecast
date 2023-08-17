WITH orders_by_day AS (
    SELECT
        formatDateTime(timestamp, '%Y-%m-%d') AS day,
        order_id,
        sku_id,
        sku,
        price,
        qty
    FROM default.demand_orders
), days_ids AS (
    SELECT day, sku_id
    FROM (SELECT DISTINCT day FROM orders_by_day) t1
    CROSS JOIN (SELECT DISTINCT sku_id FROM orders_by_day) t2
), delivered_ids AS(
    SELECT DISTINCT order_id
    FROM default.demand_orders_status
    WHERE status_id IN (1, 3, 4, 5, 6)
), sku_info AS (
    SELECT DISTINCT sku_id, sku, price
    FROM orders_by_day
), sales_qty AS (
    SELECT day, sku_id, qty
    FROM orders_by_day
    WHERE order_id IN (
        SELECT * FROM delivered_ids)
), full_sales AS (
    SELECT
        di.day,
        di.sku_id,
        sq.qty
    FROM days_ids di
    LEFT JOIN sales_qty sq
        ON di.day = sq.day
         AND di.sku_id = sq.sku_id
), sales_by_day AS (
    SELECT
        day,
        sku_id,
        SUM(qty) AS qty
    FROM full_sales
    GROUP BY day, sku_id
)
SELECT
    day,
    sd.sku_id,
    si.sku,
    si.price,
    sd.qty
FROM sales_by_day sd
LEFT JOIN sku_info si
    ON sd.sku_id = si.sku_id
ORDER BY sd.sku_id, day
