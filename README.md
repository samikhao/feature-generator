# feature-generator

API для генерации идей по feature engineering с помощью LLM.

Сервис принимает описание ML-задачи, целевой переменной, колонок датасета и ограничений, а на выходе возвращает:

- `summary` - краткое резюме
- `generated_features` - список предложенных фич
- `warnings` - риски и замечания

## Что есть в API

- `GET /health` - проверка, что сервис запущен
- `POST /generate-features` - генерация фич для ML-задачи

## Входные данные

`POST /generate-features` ожидает JSON со следующими полями:

- `project_goal` - цель модели
- `target_name` - имя таргета
- `target_type` - `binary`, `multiclass` или `regression`
- `columns` - список колонок с `name`, `dtype`, `description`
- `dataset_context` - дополнительный контекст
- `constraints` - ограничения
- `temperature` - температура генерации

## Пример запуска

```bash
uvicorn src.main:app --reload
```

## Пример использования

```bash
curl -X POST http://127.0.0.1:8000/generate-features \
  -H "Content-Type: application/json" \
  -d '{
    "project_goal": "Predict customer churn",
    "target_name": "churn",
    "target_type": "binary",
    "columns": [
      {
        "name": "age",
        "dtype": "numeric",
        "description": "Customer age"
      },
      {
        "name": "monthly_spend",
        "dtype": "numeric",
        "description": "Average monthly spend"
      },
      {
        "name": "tenure_months",
        "dtype": "numeric",
        "description": "Months since subscription start"
      }
    ],
    "dataset_context": "Telecom customer dataset",
    "constraints": [
      "avoid target leakage",
      "prefer interpretable features"
    ],
    "temperature": 0.2
  }'
```

Пример ответа:

```json
{
  "summary": "Useful features can be derived from spend, tenure and customer profile.",
  "generated_features": [
    {
      "feature_name": "spend_per_month_of_tenure",
      "source_columns": ["monthly_spend", "tenure_months"],
      "transformation": "monthly_spend / max(tenure_months, 1)",
      "description": "Normalized spend by customer lifetime.",
      "rationale": "Can separate new high-spend users from long-term low-spend users.",
      "priority": "high",
      "leakage_risk": "low"
    }
  ],
  "warnings": [
    "Check that tenure is measured before prediction time."
  ]
}
```

## Конфигурация

Сервис читает настройки из `src/.env`:

```env
API_KEY=...
API_BASE=...
MODEL_NAME=...
```
