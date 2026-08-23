-- SAE MIMIC HOURLY EXTRACTION TEMPLATE
-- IMPORTANT:
-- This is a schema/query scaffold, NOT the final Sepsis-3 + SAE cohort definition.
-- Lock the SAE onset rule and cohort definition before using it for results.
--
-- Expected output:
-- subject_id, stay_id, hour, sae_onset_hour,
-- heart_rate, map, resp_rate, spo2, gcs
--
-- Adapt table names / itemids to the exact MIMIC-IV version and your approved
-- cohort definition.

WITH hourly AS (
  SELECT
    subject_id,
    stay_id,
    CAST(FLOOR((charttime - intime) / 3600) AS INT64) AS hour,

    -- Replace these placeholders with validated MIMIC-IV item mappings.
    AVG(CASE WHEN itemid IN (<HEART_RATE_ITEMIDS>) THEN valuenum END) AS heart_rate,
    AVG(CASE WHEN itemid IN (<MAP_ITEMIDS>) THEN valuenum END) AS map,
    AVG(CASE WHEN itemid IN (<RESP_RATE_ITEMIDS>) THEN valuenum END) AS resp_rate,
    AVG(CASE WHEN itemid IN (<SPO2_ITEMIDS>) THEN valuenum END) AS spo2,
    AVG(CASE WHEN itemid IN (<GCS_ITEMIDS>) THEN valuenum END) AS gcs

  FROM `<PROJECT>.<DATASET>.chartevents`
  WHERE valuenum IS NOT NULL
  GROUP BY subject_id, stay_id, hour
)

SELECT *
FROM hourly
ORDER BY subject_id, stay_id, hour;
