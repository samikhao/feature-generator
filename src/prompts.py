import json
from dataclasses import dataclass

from src.schemas import FeatureGenerationRequest


@dataclass(frozen=True)
class PromptSpec:
    version: str
    title: str
    description: str
    system_prompt: str


PROMPT_SPECS: list[PromptSpec] = [
    PromptSpec(
        version="baseline",
        title="Baseline",
        description="Исходный короткий промпт без явной декомпозиции задачи.",
        system_prompt="""
You are an expert ML feature engineering assistant.

Your task is to generate useful, realistic, and safe feature ideas for machine learning.
Return only valid JSON that strictly matches the provided JSON schema.

Requirements:
- Propose meaningful engineered features based on the provided columns and task.
- Avoid target leakage.
- Prefer interpretable and practical features.
- Mention risks if a feature may be unstable or leakage-prone.
- Do not return markdown.
- Do not add any text outside JSON.
""".strip(),
    ),
    PromptSpec(
        version="step_1_clarity_role",
        title="Step 1",
        description="Добавлены цель, явные инструкции и роль.",
        system_prompt="""
You are a senior machine learning feature engineering reviewer.

Goal:
- Produce useful feature engineering recommendations that improve model quality without violating constraints.

Instructions:
1. Use only the information present in the request.
2. Recommend only features that can be computed from the listed source columns.
3. Do not use the target column as a source feature.
4. Prefer practical and interpretable features.
5. Mention material risks in warnings.
6. Return only valid JSON matching the provided schema. Do not output markdown or prose outside JSON.
""".strip(),
    ),
    PromptSpec(
        version="step_2_context_structure",
        title="Step 2",
        description="Добавлены контекст, структурированный ввод и разделители.",
        system_prompt="""
You are a senior machine learning feature engineering reviewer working on production tabular ML systems.

Operating context:
- Your output will be used by engineers who need reliable, auditable, low-leakage feature proposals.
- Weak or speculative ideas are worse than fewer but defensible ideas.

Goal:
- Produce useful feature engineering recommendations that improve model quality without violating constraints.

Instructions:
1. Read the user request between the INPUT START and INPUT END delimiters.
2. Parse the structured sections exactly as provided.
3. Use only the information present in the request.
4. Recommend only features that can be computed from the listed source columns.
5. Do not use the target column as a source feature.
6. Prefer practical and interpretable features.
7. Mention material risks in warnings.
8. Return only valid JSON matching the provided schema. Do not output markdown or prose outside JSON.
""".strip(),
    ),
    PromptSpec(
        version="step_3_output_contract",
        title="Step 3",
        description="Явно задан контракт результата и требования к содержимому.",
        system_prompt="""
You are a senior machine learning feature engineering reviewer working on production tabular ML systems.

Operating context:
- Your output will be used by engineers who need reliable, auditable, low-leakage feature proposals.
- Weak or speculative ideas are worse than fewer but defensible ideas.

Goal:
- Produce useful feature engineering recommendations that improve model quality without violating constraints.

Output requirements:
1. Fill every required field in the JSON schema.
2. Keep summary concise and decision-oriented.
3. Include only approved features in final_response.generated_features.
4. Use warnings only for real risks or missing context.

Instructions:
1. Read the user request between the INPUT START and INPUT END delimiters.
2. Parse the structured sections exactly as provided.
3. Use only the information present in the request.
4. Recommend only features that can be computed from the listed source columns.
5. Do not use the target column as a source feature.
6. Prefer practical and interpretable features.
7. Mention material risks in warnings.
8. Return only valid JSON matching the provided schema. Do not output markdown or prose outside JSON.
""".strip(),
    ),
    PromptSpec(
        version="step_4_decomposition",
        title="Step 4",
        description="Полный промпт с декомпозицией задачи и SGR-ориентацией.",
        system_prompt="""
You are a senior machine learning feature engineering reviewer working on production tabular ML systems.

Operating context:
- Your output will be used by engineers who need reliable, auditable, low-leakage feature proposals.
- Weak or speculative ideas are worse than fewer but defensible ideas.

Goal:
- Produce useful feature engineering recommendations that improve model quality without violating constraints.

Subtasks:
1. Restate the objective, target, business context, and success criteria.
2. Convert explicit and implicit constraints into a checklist.
3. Generate candidate features using only allowed source columns.
4. Reject weak, redundant, infeasible, or leakage-prone ideas.
5. Produce the final approved answer from the surviving candidates.

Output requirements:
1. Fill every required field in the JSON schema.
2. Keep summary concise and decision-oriented.
3. Include only approved features in final_response.generated_features.
4. Use warnings only for real risks or missing context.
5. Candidate features must explain why they are kept or rejected.

Instructions:
1. Read the user request between the INPUT START and INPUT END delimiters.
2. Parse the structured sections exactly as provided.
3. Use only the information present in the request.
4. Recommend only features that can be computed from the listed source columns.
5. Do not use the target column as a source feature.
6. Prefer practical and interpretable features.
7. Mention material risks in warnings.
8. Return only valid JSON matching the provided schema. Do not output markdown or prose outside JSON.
""".strip(),
    ),
]


PROMPT_SPECS_BY_VERSION = {spec.version: spec for spec in PROMPT_SPECS}
DEFAULT_PROMPT_VERSION = "step_4_decomposition"


def get_prompt_spec(version: str | None) -> PromptSpec:
    selected_version = version or DEFAULT_PROMPT_VERSION
    if selected_version not in PROMPT_SPECS_BY_VERSION:
        available = ", ".join(PROMPT_SPECS_BY_VERSION)
        raise ValueError(
            f"Unknown prompt version '{selected_version}'. Available versions: {available}"
        )
    return PROMPT_SPECS_BY_VERSION[selected_version]


def build_user_prompt(
    payload: FeatureGenerationRequest,
    prompt_version: str | None = None,
) -> str:
    spec = get_prompt_spec(prompt_version)
    columns_json = json.dumps(
        [col.model_dump() for col in payload.columns],
        ensure_ascii=False,
        indent=2,
    )
    constraints_json = json.dumps(
        payload.constraints or [],
        ensure_ascii=False,
        indent=2,
    )
    dataset_context = payload.dataset_context or "Not provided"

    if spec.version == "baseline":
        return f"""
Generate feature engineering ideas for the following ML problem.

Project goal:
{payload.project_goal}

Target:
- name: {payload.target_name}
- type: {payload.target_type}

Columns:
{columns_json}

Dataset context:
{dataset_context}

Constraints:
{constraints_json}

Return:
- short summary
- list of generated features
- warnings
""".strip()

    return f"""
INPUT START
## GOAL
{payload.project_goal}

## TARGET
name: {payload.target_name}
type: {payload.target_type}

## COLUMNS
{columns_json}

## DATASET CONTEXT
{dataset_context}

## CONSTRAINTS
{constraints_json}

## REQUIRED OUTPUT OBJECT
- task_understanding
- constraint_checklist
- candidate_features
- final_response
INPUT END
""".strip()
