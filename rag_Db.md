# Feature Engineering Knowledge Base for Metadata-Based RAG

Purpose: This knowledge base contains reusable feature engineering rules and patterns for tabular machine learning tasks. It is designed for a RAG system that receives metadata only: project goal, target name, target type, column names, dtypes, column descriptions, dataset context, and constraints. It does not assume access to raw data values, distributions, missing rates, correlations, target statistics, or sample rows.

Use this knowledge base to generate feature ideas, not preprocessing steps. Do not suggest using raw columns as engineered features. Every suggested feature should transform, combine, aggregate, compare, bucket, encode semantically, or derive new information from existing columns.

---

## Global Rule: Avoid Identity Features

Do not suggest a feature that is simply the original column copied into the model.

Bad examples:
- use `age` as a feature
- use `income` as a feature
- use `country` as a feature

Good examples:
- `age_group`
- `income_to_debt_ratio`
- `is_domestic_country`
- `days_since_signup`
- `customer_lifetime_bucket`

A feature idea should add information through transformation, combination, grouping, comparison, or domain reasoning.

---

## Global Rule: Respect Target Leakage Constraints

Avoid features that use information unavailable at prediction time.

Common leakage risks:
- columns created after the target event
- future transactions
- final statuses
- post-outcome dates
- manually assigned resolution labels
- direct proxies of the target

For temporal or event-based datasets, prefer historical features based only on information available before the prediction timestamp.

Safe phrasing:
- historical count before prediction time
- rolling average over prior events
- days since last known event
- number of previous interactions

Unsafe phrasing:
- total future spend
- final account status
- days until churn
- outcome-confirming flag

---

## Global Rule: Match Feature Ideas to Target Type

For binary classification tasks, useful engineered features often express risk, propensity, similarity, intensity, change, flags, ratios, and behavioral patterns.

For multiclass classification tasks, useful engineered features often express class-specific behavior, group membership, categorical combinations, and domain-specific segments.

For regression tasks, useful engineered features often express magnitude, normalization, ratios, trends, seasonality, totals, rates, and intensity.

Do not force classification-only features into regression tasks, or regression-only scoring features into classification tasks.

---

## Rule: Numeric Ratio Features

Use ratio features when two numeric columns describe quantities where one can normalize the other.

Useful patterns:
- amount per unit
- price per area
- spend per order
- debt to income
- clicks per impression
- revenue per customer
- duration per event
- cost per item

Example feature ideas:
- `price_per_area = price / area`
- `debt_to_income_ratio = debt / income`
- `avg_order_value = total_spend / order_count`
- `conversion_rate = purchases / visits`

Avoid ratio features when the denominator may be zero, unstable, or semantically unrelated. If the data is not available, phrase the idea conditionally: "if denominator values are valid".

---

## Rule: Numeric Difference Features

Use difference features when two columns represent comparable quantities, states, balances, dates, scores, or measurements.

Useful patterns:
- current minus previous
- expected minus actual
- requested minus approved
- start minus end
- maximum minus minimum
- income minus expenses

Example feature ideas:
- `income_expense_gap = income - monthly_expenses`
- `requested_approved_gap = requested_amount - approved_amount`
- `score_change = current_score - previous_score`
- `price_discount_gap = original_price - final_price`

Difference features are useful when the gap itself has domain meaning.

---

## Rule: Numeric Product and Interaction Features

Use product features when two numeric values together represent combined intensity, exposure, or scale.

Useful patterns:
- quantity times price
- duration times rate
- frequency times amount
- probability times value

Example feature ideas:
- `estimated_total_cost = unit_price * quantity`
- `usage_intensity = session_duration * sessions_count`
- `risk_exposure = risk_score * loan_amount`
- `engagement_volume = visit_count * average_duration`

Product features should be interpretable and domain-relevant.

---

## Rule: Bucketed Numeric Features

Convert continuous numeric columns into meaningful groups when ranges have domain interpretation.

Useful patterns:
- age groups
- income bands
- risk score bands
- account tenure groups
- transaction amount tiers
- usage intensity levels

Example feature ideas:
- `age_group`
- `income_band`
- `loan_amount_tier`
- `tenure_bucket`
- `risk_score_segment`

Bucketed features are useful for interpretability and for capturing non-linear effects.

Avoid arbitrary buckets without explanation. Prefer domain-relevant thresholds when possible.

---

## Rule: Logarithmic Magnitude Features

For positive numeric columns that represent money, counts, size, duration, or volume, consider log-transformed magnitude features.

Useful patterns:
- log transaction amount
- log income
- log order count
- log account balance
- log page views

Example feature ideas:
- `log_income`
- `log_transaction_amount`
- `log_total_spend`
- `log_order_count`

This is a feature transformation, not scaling. It may help represent multiplicative effects and reduce dominance of very large values.

Use only when values are non-negative or can be safely shifted.

---

## Rule: Boolean Flag Features

Create boolean flags from numeric, categorical, temporal, or text metadata when a simple condition has domain meaning.

Useful patterns:
- has previous activity
- is high value
- is new customer
- is weekend
- is business hour
- is international
- has discount
- has support contact

Example feature ideas:
- `is_new_customer`
- `has_prior_purchase`
- `is_high_value_order`
- `is_weekend_event`
- `is_international_transaction`
- `has_active_subscription`

Boolean flags are useful when the condition is easy to interpret and likely relevant to the target.

---

## Rule: Count Features

Use count features when the dataset contains entities, events, transactions, sessions, interactions, products, messages, or records that can be grouped.

Useful patterns:
- number of orders per customer
- number of transactions per account
- number of logins per user
- number of support tickets per customer
- number of products per order
- number of failed attempts per user

Example feature ideas:
- `customer_order_count`
- `account_transaction_count`
- `user_login_count`
- `support_ticket_count`
- `failed_payment_count`

Counts should be based only on historical data available at prediction time.

---

## Rule: Frequency and Rate Features

Use frequency or rate features when counts can be normalized by time, exposure, opportunity, or entity size.

Useful patterns:
- orders per month
- logins per week
- transactions per day
- complaints per order
- clicks per impression
- purchases per visit

Example feature ideas:
- `orders_per_month`
- `logins_per_week`
- `transactions_per_day`
- `complaints_per_order`
- `purchase_rate`

Rate features are often more meaningful than raw counts because they normalize activity level.

---

## Rule: Aggregation Features by Entity

If the dataset has entity identifiers such as customer_id, user_id, account_id, product_id, merchant_id, device_id, or store_id, suggest historical aggregation features for that entity.

Useful aggregations:
- count
- sum
- mean
- median
- min
- max
- standard deviation
- last value
- first value
- number of unique values

Example feature ideas:
- `customer_avg_order_value`
- `user_total_sessions`
- `account_max_transaction_amount`
- `merchant_unique_customer_count`
- `product_avg_rating`

Aggregation features should be computed using only past records relative to the prediction moment when time is involved.

---

## Rule: Group Comparison Features

Create features that compare an entity to a broader group, segment, or population.

Useful patterns:
- customer value relative to average customer value
- transaction amount relative to merchant average
- product price relative to category average
- user activity relative to regional average

Example feature ideas:
- `transaction_amount_vs_customer_avg`
- `order_value_vs_category_avg`
- `customer_spend_vs_segment_avg`
- `product_price_vs_brand_avg`

These features express whether something is unusually high or low compared with a relevant baseline.

Avoid using target-derived group statistics unless computed safely without leakage.

---

## Rule: Temporal Date Part Features

If columns represent dates or timestamps, derive calendar-based features.

Useful patterns:
- hour of day
- day of week
- day of month
- month
- quarter
- year
- weekend flag
- business hours flag
- holiday period flag
- season

Example feature ideas:
- `signup_month`
- `transaction_day_of_week`
- `event_hour`
- `is_weekend_transaction`
- `is_business_hours_event`
- `purchase_quarter`

Calendar features are useful when behavior depends on time cycles.

---

## Rule: Temporal Recency Features

If the dataset has timestamps or event dates, derive recency features that measure elapsed time since an important event.

Useful patterns:
- days since signup
- days since last purchase
- days since last login
- days since last support ticket
- days since account update
- time since previous transaction

Example feature ideas:
- `days_since_signup`
- `days_since_last_purchase`
- `days_since_last_login`
- `days_since_last_support_contact`
- `time_since_previous_event`

Recency features are useful for churn, fraud, engagement, retention, conversion, and lifecycle modeling.

Do not use future events after the prediction point.

---

## Rule: Temporal Duration Features

Use duration features when the dataset contains start and end dates, opening and closing dates, activation and cancellation dates, or event start and finish timestamps.

Useful patterns:
- account age
- subscription duration
- session duration
- processing time
- delivery time
- time from application to approval

Example feature ideas:
- `account_age_days`
- `subscription_duration_days`
- `session_duration_minutes`
- `application_processing_time_days`
- `delivery_duration_days`

Duration features should only use end dates that are available at prediction time.

---

## Rule: Temporal Trend Features

If historical events or repeated measurements exist, derive trend features.

Useful patterns:
- increasing spend
- decreasing usage
- change in frequency
- recent average versus earlier average
- current value versus previous value

Example feature ideas:
- `spend_trend_last_3_months`
- `usage_change_recent_vs_previous_period`
- `login_frequency_trend`
- `average_order_value_change`

Trend features are useful when direction of behavior matters.

Trends must be computed from historical periods only.

---

## Rule: Rolling Window Features

For event-based or time-indexed datasets, suggest rolling historical window features.

Useful windows:
- last 7 days
- last 30 days
- last 90 days
- previous month
- previous quarter

Useful aggregations:
- count
- sum
- average
- maximum
- minimum
- standard deviation
- unique count

Example feature ideas:
- `transaction_count_last_30_days`
- `total_spend_last_90_days`
- `avg_session_duration_last_7_days`
- `unique_merchants_last_30_days`

Rolling window features are often strong for churn, fraud, financial risk, demand forecasting, and engagement tasks.

Avoid windows that include the target event or future events.

---

## Rule: Categorical Combination Features

Combine two or more categorical columns when the interaction between categories may be more informative than each category separately.

Useful patterns:
- country + device type
- product category + region
- merchant category + payment method
- customer segment + channel
- plan type + acquisition source

Example feature ideas:
- `country_device_combo`
- `product_category_region_combo`
- `merchant_category_payment_method_combo`
- `segment_channel_combo`

Categorical combinations are useful when behavior differs across category intersections.

Avoid overly high-cardinality combinations unless controlled.

---

## Rule: Semantic Category Grouping

Group detailed categorical values into broader semantic groups when the original categories may be too granular or noisy.

Useful patterns:
- job title to seniority level
- product to product family
- country to region
- transaction type to transaction group
- device model to device brand
- error code to error family

Example feature ideas:
- `country_region`
- `job_seniority_level`
- `product_family`
- `transaction_type_group`
- `device_brand`

This type of feature is especially useful when column descriptions imply hierarchical or domain categories.

---

## Rule: Text Metadata Length Features

If a column contains free text, names, descriptions, comments, reviews, messages, or titles, derive simple text structure features.

Useful patterns:
- text length
- word count
- number of sentences
- number of digits
- number of special characters
- uppercase ratio
- contains URL
- contains email

Example feature ideas:
- `review_word_count`
- `message_length_chars`
- `description_sentence_count`
- `contains_url_flag`
- `uppercase_ratio`

These features are not NLP embeddings; they are simple engineered metadata features.

---

## Rule: Text Keyword Flags

If text columns are domain-relevant, suggest keyword or pattern flags based on business meaning.

Useful patterns:
- complaint keywords
- urgency words
- cancellation intent
- refund mentions
- fraud-related terms
- positive or negative sentiment indicators

Example feature ideas:
- `contains_refund_keyword`
- `contains_cancel_keyword`
- `contains_urgent_keyword`
- `contains_error_keyword`
- `contains_fraud_claim_keyword`

Keyword features should be interpretable and aligned with the project goal.

---

## Rule: Identifier-Derived Features

Identifier columns should usually not be used directly. However, they may enable grouping, counting, uniqueness, or history-based features.

Useful identifier columns:
- user_id
- customer_id
- account_id
- merchant_id
- product_id
- order_id
- device_id
- session_id

Good feature ideas:
- `customer_order_count`
- `user_unique_device_count`
- `merchant_transaction_count`
- `account_days_since_first_seen`
- `product_purchase_count`

Bad feature ideas:
- use `customer_id` directly
- use `order_id` directly
- encode arbitrary IDs without justification

Identifiers are keys for aggregation, not usually predictive features themselves.

---

## Rule: Unique Count Features

Use unique count features when an entity interacts with many distinct objects.

Useful patterns:
- number of unique merchants per customer
- number of unique products per user
- number of unique devices per account
- number of unique locations per user
- number of unique categories per order

Example feature ideas:
- `unique_merchants_per_customer`
- `unique_products_per_user`
- `unique_devices_per_account`
- `unique_locations_per_user`
- `unique_categories_per_customer`

Unique count features often represent diversity, complexity, suspicious behavior, or engagement breadth.

---

## Rule: First and Last Event Features

If event history exists, derive features based on first or last known event.

Useful patterns:
- first purchase date
- last purchase date
- first login date
- last login date
- first transaction amount
- last transaction amount
- last status before prediction

Example feature ideas:
- `days_since_first_purchase`
- `days_since_last_purchase`
- `last_transaction_amount`
- `first_order_value`
- `last_known_status_before_prediction`

Only use events available before prediction time.

---

## Rule: Rank and Percentile Features

Suggest rank or percentile features when an entity, product, account, or transaction can be compared within a group.

Useful patterns:
- transaction amount percentile within customer history
- product price rank within category
- customer spend rank within segment
- order size percentile within store

Example feature ideas:
- `transaction_amount_percentile_for_customer`
- `product_price_rank_in_category`
- `customer_spend_percentile_in_region`
- `order_value_rank_for_merchant`

Rank features are useful for detecting unusually high or low values relative to context.

---

## Rule: Change From Baseline Features

Create features that compare a current value to a historical baseline.

Useful patterns:
- current spend versus average spend
- current login frequency versus previous frequency
- latest score versus historical average
- current transaction amount versus usual amount

Example feature ideas:
- `current_spend_vs_avg_spend`
- `latest_score_vs_historical_avg_score`
- `current_transaction_vs_customer_avg`
- `recent_usage_vs_baseline_usage`

These features are strong for anomaly, churn, fraud, and behavior change tasks.

---

## Rule: Customer Churn Feature Patterns

For churn prediction, prioritize features that describe engagement, recency, lifecycle, usage decline, support interactions, payment issues, and customer value.

Useful feature ideas:
- `days_since_last_login`
- `days_since_last_purchase`
- `usage_trend_last_30_days`
- `support_ticket_count_last_90_days`
- `payment_failure_count`
- `subscription_age_days`
- `plan_change_count`
- `downgrade_flag`
- `avg_monthly_spend`
- `recent_activity_vs_previous_activity`

Avoid features that are only known after churn occurs, such as cancellation reason or final account status, unless they are available before prediction.

---

## Rule: Fraud Detection Feature Patterns

For fraud detection, prioritize features that describe unusual behavior, velocity, device changes, location changes, transaction patterns, identity consistency, and deviation from history.

Useful feature ideas:
- `transaction_count_last_1_hour`
- `transaction_amount_vs_customer_avg`
- `unique_devices_last_30_days`
- `unique_locations_last_7_days`
- `is_new_device`
- `is_new_merchant_for_customer`
- `failed_attempt_count_last_24_hours`
- `distance_from_usual_location`
- `amount_percentile_for_customer`

Avoid using investigation outcomes, chargeback labels, or post-event fraud decisions as features.

---

## Rule: Credit Risk Feature Patterns

For credit scoring, loan default, and financial risk tasks, prioritize features that describe affordability, leverage, repayment behavior, stability, and historical risk.

Useful feature ideas:
- `debt_to_income_ratio`
- `monthly_payment_to_income_ratio`
- `loan_amount_to_income_ratio`
- `credit_utilization_ratio`
- `income_expense_gap`
- `employment_duration_bucket`
- `previous_default_count`
- `late_payment_count`
- `account_age_days`
- `requested_amount_vs_income`

Avoid using future repayment outcomes or post-approval information when predicting approval/default risk.

---

## Rule: E-Commerce Feature Patterns

For e-commerce tasks such as conversion, purchase prediction, demand, recommendation, or customer value, prioritize features that describe browsing, purchase history, basket composition, product affinity, discounts, and recency.

Useful feature ideas:
- `days_since_last_purchase`
- `customer_order_count`
- `avg_order_value`
- `cart_item_count`
- `discount_usage_rate`
- `product_category_diversity`
- `repeat_purchase_flag`
- `views_to_purchases_ratio`
- `wishlist_to_purchase_flag`
- `recent_spend_vs_lifetime_avg_spend`

Avoid features based on events that happen after the prediction decision, such as final purchase status when predicting conversion.

---

## Rule: Marketing and Lead Scoring Feature Patterns

For lead scoring, campaign response, conversion, or marketing propensity tasks, prioritize engagement, channel, recency, frequency, campaign interaction, and customer profile features.

Useful feature ideas:
- `email_open_rate`
- `click_to_open_ratio`
- `days_since_last_campaign_interaction`
- `campaign_interaction_count`
- `preferred_channel`
- `lead_age_days`
- `form_completion_depth`
- `number_of_touchpoints`
- `recent_engagement_score`
- `channel_campaign_combo`

Avoid using post-conversion information when predicting conversion likelihood.

---

## Rule: Healthcare Feature Patterns

For healthcare tabular tasks, prioritize features that describe patient history, recency of events, clinically meaningful ratios, counts of visits, medication history, and temporal disease progression.

Useful feature ideas:
- `days_since_last_visit`
- `visit_count_last_12_months`
- `medication_count`
- `condition_count`
- `lab_value_change_from_previous`
- `body_mass_index_if_height_weight_available`
- `age_group`
- `hospitalization_count_last_year`
- `comorbidity_count`

Avoid features that use information recorded after diagnosis, discharge, or outcome confirmation when predicting that outcome.

---

## Rule: Real Estate Feature Patterns

For real estate pricing, valuation, demand, or risk tasks, prioritize normalized price, size, location, age, density, and amenity-derived features.

Useful feature ideas:
- `price_per_square_meter`
- `rooms_per_area`
- `property_age_years`
- `distance_to_city_center_bucket`
- `floor_ratio = floor / total_floors`
- `has_parking_flag`
- `has_balcony_flag`
- `location_property_type_combo`
- `area_per_room`

Avoid using future sale price changes when predicting current price or demand.

---

## Rule: Subscription Product Feature Patterns

For subscription products, prioritize lifecycle, plan changes, billing behavior, engagement, support, and usage trends.

Useful feature ideas:
- `subscription_age_days`
- `days_until_renewal`
- `days_since_last_active_use`
- `plan_change_count`
- `downgrade_flag`
- `payment_failure_count`
- `support_ticket_count_last_90_days`
- `usage_trend_last_30_days`
- `feature_adoption_count`

Avoid using cancellation date or cancellation reason if predicting churn before cancellation.

---

## Rule: Operational and Logistics Feature Patterns

For delivery, logistics, supply chain, or operations tasks, prioritize duration, delays, capacity, distance, route complexity, workload, and historical performance.

Useful feature ideas:
- `delivery_duration_hours`
- `order_processing_time`
- `distance_per_delivery`
- `items_per_shipment`
- `delay_from_expected_time`
- `carrier_avg_delay`
- `warehouse_order_volume_last_7_days`
- `route_complexity_score`
- `is_peak_period`

Avoid using actual delivery completion time when predicting delay before delivery is completed.

---

## Rule: Education Feature Patterns

For education, student performance, dropout, or engagement tasks, prioritize attendance, assignment history, activity, recency, consistency, and progress.

Useful feature ideas:
- `attendance_rate`
- `assignment_completion_rate`
- `days_since_last_login`
- `late_submission_count`
- `average_score_trend`
- `course_progress_percent`
- `forum_activity_count`
- `study_session_count_last_30_days`
- `missed_deadline_count`

Avoid using final grades or completion status when predicting early performance or dropout risk.

---

## Rule: Manufacturing and IoT Feature Patterns

For manufacturing, sensor, maintenance, or IoT tasks, prioritize rolling statistics, deviations, operating duration, counts of alerts, recent changes, and equipment age.

Useful feature ideas:
- `machine_age_days`
- `operating_hours_since_last_maintenance`
- `sensor_value_rolling_mean`
- `sensor_value_rolling_std`
- `alert_count_last_7_days`
- `temperature_change_from_previous_reading`
- `current_value_vs_machine_baseline`
- `maintenance_count_last_year`

Avoid using failure confirmation or repair outcome when predicting failure risk.

---

## Rule: Financial Transaction Feature Patterns

For banking and transaction tasks, prioritize amount normalization, velocity, merchant behavior, account history, balance changes, and transaction diversity.

Useful feature ideas:
- `transaction_amount_vs_account_avg`
- `transaction_count_last_24_hours`
- `unique_merchants_last_30_days`
- `balance_change_after_transaction`
- `merchant_category_transaction_count`
- `cash_withdrawal_ratio`
- `international_transaction_flag`
- `weekend_transaction_flag`
- `night_transaction_flag`

Avoid post-transaction investigation or final dispute status when predicting risk at transaction time.

---

## Rule: Geographical Feature Patterns

If the dataset contains location, country, region, city, latitude, longitude, address, or store location columns, derive geographical features.

Useful feature ideas:
- `country_region_group`
- `is_domestic_location`
- `distance_to_home_location`
- `distance_to_nearest_store`
- `location_change_flag`
- `unique_locations_per_user`
- `city_tier`
- `same_country_as_customer_flag`

Use distance and location-change features when multiple location fields exist.

Avoid overly specific location identifiers when privacy or generalization is important.

---

## Rule: Hierarchical Category Features

If columns imply hierarchy, derive parent-level or level-based features.

Useful hierarchies:
- product -> category -> department
- city -> region -> country
- job title -> role family -> seniority
- error code -> error group
- merchant -> merchant category

Example feature ideas:
- `product_parent_category`
- `city_region`
- `job_seniority`
- `error_code_family`
- `merchant_category_group`

Hierarchical features improve generalization when fine-grained categories are sparse.

---

## Rule: Consistency Check Features

Create features that compare related fields for internal consistency.

Useful patterns:
- billing country equals shipping country
- declared income consistent with occupation
- transaction location matches user country
- device country matches account country
- order currency matches customer region

Example feature ideas:
- `billing_shipping_country_match_flag`
- `device_country_matches_account_country`
- `transaction_country_matches_home_country`
- `currency_matches_region_flag`

Consistency features are useful for fraud, risk, data quality, and identity verification tasks.

---

## Rule: Lifecycle Stage Features

Create lifecycle features when columns describe signup, activation, tenure, usage, renewal, cancellation, or maturity.

Useful feature ideas:
- `customer_lifecycle_stage`
- `account_age_bucket`
- `days_since_activation`
- `new_customer_flag`
- `mature_account_flag`
- `renewal_period_flag`
- `early_lifecycle_activity_count`

Lifecycle features are useful for churn, conversion, risk, retention, and customer value modeling.

---

## Rule: Intensity Features

Create intensity features when raw volume should be normalized by duration, size, count, or opportunity.

Useful feature ideas:
- `spend_per_active_day`
- `events_per_session`
- `usage_minutes_per_day`
- `tickets_per_customer_age_month`
- `transactions_per_account_age_day`
- `views_per_product_age_day`

Intensity features help distinguish active long-term entities from recently active short-term entities.

---

## Rule: Diversity Features

Create diversity features when an entity can interact with multiple categories, products, locations, devices, merchants, or channels.

Useful feature ideas:
- `unique_product_categories_count`
- `unique_merchant_categories_count`
- `unique_channels_used`
- `unique_device_count`
- `unique_locations_count`
- `category_diversity_ratio`

Diversity features can represent exploration, complexity, suspiciousness, or engagement breadth.

---

## Rule: Stability and Volatility Features

If repeated numeric measurements exist, derive stability or volatility features.

Useful feature ideas:
- `transaction_amount_std_last_30_days`
- `balance_volatility_last_90_days`
- `usage_std_last_4_weeks`
- `score_variability`
- `order_value_volatility`

Volatility features are useful for risk, fraud, forecasting, churn, and anomaly detection.

---

## Rule: Missingness Indicator Feature Ideas

If profiling later shows missing values, missingness itself can be represented as a feature.

Useful feature ideas:
- `income_missing_flag`
- `last_login_missing_flag`
- `phone_number_missing_flag`
- `address_missing_flag`
- `profile_completion_flag`

In the current metadata-only mode, suggest these conditionally: "if this column has missing values".

Do not claim missingness exists unless profiling data is available.

---

## Rule: Profile Completeness Features

If several optional fields exist, derive completeness features.

Useful feature ideas:
- `profile_completed_field_count`
- `profile_completion_ratio`
- `has_contact_information_flag`
- `has_verified_identity_flag`
- `missing_required_info_count`

These are useful for fraud, conversion, churn, onboarding, and customer quality tasks.

---

## Rule: Target Encoding Caution

Target encoding is risky because it can cause target leakage if not computed with proper cross-validation or historical-only logic.

In metadata-only ideation, avoid directly recommending target encoding unless the response includes a leakage-safe condition.

Safe phrasing:
- `category_historical_target_rate` computed out-of-fold or from past data only
- `merchant_historical_fraud_rate` computed only using prior transactions

Unsafe phrasing:
- use target mean for each category without validation safeguards

For the current system, prefer semantic category grouping and interaction features over target encoding unless leakage controls are explicit.

---

## Rule: Production Constraints

Avoid feature ideas that would be hard to compute reliably in production unless clearly marked as optional or offline-only.

Risky production features:
- features requiring future data
- features requiring manual labels
- expensive external API calls
- unstable free-text parsing
- features unavailable at prediction time
- features requiring full dataset recomputation for each prediction

Prefer feature ideas that can be computed from known input columns, historical tables, or stable business logic.

---

## Rule: Interpretability Constraints

When interpretability is important, prefer simple and explainable features.

Good interpretable features:
- ratios
- counts
- flags
- buckets
- recency
- duration
- differences
- group comparisons

Less interpretable features:
- high-degree polynomial features
- opaque embeddings
- complex learned representations
- arbitrary hashed combinations

If constraints mention interpretability, prioritize features that a domain expert can understand.

---

## Rule: Feature Naming

Generated feature names should be concise, descriptive, and implementation-friendly.

Good names:
- `days_since_last_purchase`
- `debt_to_income_ratio`
- `transaction_count_last_30_days`
- `customer_order_count`
- `is_weekend_transaction`

Bad names:
- `feature_1`
- `enhanced_customer_behavior_metric`
- `use age better`
- `derived thing from date`

Prefer snake_case names.

---

## Rule: Feature Idea Explanation

Each feature idea should explain:
- what columns it uses
- how it is derived
- why it may help the target
- leakage or production risks if relevant

Example:
`days_since_last_purchase`: derived from customer purchase timestamp history; may help churn prediction because inactive customers are more likely to churn; must use only purchases before prediction time.

---

## Rule: Metadata-Only Honesty

When only metadata is available, do not claim that a feature is definitely predictive.

Use cautious wording:
- may help
- could capture
- useful candidate
- worth testing
- if this column exists and is available at prediction time

Avoid unsupported claims:
- this feature will improve accuracy
- this column has high correlation
- this feature is important
- this distribution is skewed

---

## Rule: Recommended Output Diversity

For each request, try to generate a diverse set of feature ideas across several categories when applicable:
- temporal features
- ratio features
- difference features
- aggregation features
- count or frequency features
- categorical interaction features
- semantic grouping features
- boolean flag features
- lifecycle features
- leakage-safe historical features

Avoid returning many variants of the same idea unless the user asks for exhaustive suggestions.

---

## Rule: When Not to Suggest a Feature

Reject or warn about feature ideas when:
- the feature is just a raw column
- the feature directly leaks the target
- the feature requires data not available at prediction time
- the feature is impossible from provided columns
- the feature is preprocessing rather than feature engineering
- the feature cannot be explained
- the feature conflicts with constraints

Add such ideas to rejected_feature_ideas in audit when the response schema supports it.

---

## Rule: Metadata Query Expansion for Retrieval

When retrieving from this knowledge base, use the project goal, target type, dataset context, column names, dtypes, and column descriptions.

Useful retrieval query terms:
- target domain: churn, fraud, credit risk, pricing, demand, conversion, healthcare, logistics
- column types: numeric, categorical, datetime, text, id
- semantic column names: amount, price, income, date, timestamp, user_id, customer_id, status, location
- constraints: leakage, interpretability, production, temporal, privacy

Retrieve both global rules and domain-specific rules.

---

## Rule: Suggested RAG Prompt Integration

When using retrieved rules, instruct the model to follow them explicitly.

Recommended instruction:
Use the retrieved feature engineering rules as grounding. Generate feature ideas only if they are supported by the provided metadata and do not violate constraints. Do not suggest preprocessing-only steps. Do not suggest identity features. If a feature idea has leakage risk, include a warning or reject it.

---

## Rule: Audit Metadata for RAG

If the system uses RAG, include retrieval information in the audit when possible.

Useful audit fields:
- retrieved_rule_ids
- retrieved_domains
- retrieval_query
- rejected_feature_ideas
- leakage_warnings
- production_warnings

This improves transparency and helps evaluate whether RAG actually influenced the answer.

