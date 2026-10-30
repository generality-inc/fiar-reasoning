"""
LLM names used in the tournament.

API_TO_NAME maps the model identifier recorded in the game logs (with the
reasoning-effort suffix, as in the 'model_names' column of
data/game_trees_df_annotated.pkl) to the name used in the paper.
"""

API_TO_NAME = {
    # trace-accessible (paper Appendix B)
    'accounts/fireworks/models/deepseek-r1-0528-medium': 'deepseek-r1',
    'deepseek-ai/deepseek-v3.1-maas': 'deepseek-v3.1',
    'accounts/fireworks/models/deepseek-v3p1-medium': 'deepseek-v3.1-thinking',
    'deepseek-reasoner': 'deepseek-v3.2-thinking',
    'accounts/fireworks/models/glm-4p5-medium': 'glm-4.5-thinking',
    'glm-4.6': 'glm-4.6',
    'accounts/fireworks/models/kimi-k2-instruct-0905': 'kimi-k2-instruct',
    'kimi-k2-thinking': 'kimi-k2-thinking',
    'accounts/fireworks/models/llama-v3p3-70b-instruct': 'llama-v3.3-70b-instruct',
    'accounts/fireworks/models/qwen3-235b-a22b-thinking-2507-medium': 'qwen3-235b-a22b-thinking',
    'qwen3-max': 'qwen3-max',
    'qwen/qwen3-next-80b-a3b-thinking-maas': 'qwen3-next-80b-a3b-thinking',
    'openai/gpt-oss-120b-maas-medium': 'gpt-oss-120b-medium',
    'openai/gpt-oss-120b-maas-high': 'gpt-oss-120b-high',
    # proprietary (reasoning summaries only)
    'gpt-4o-2024-08-06': 'gpt-4o',
    'gpt-5-2025-08-07-medium': 'gpt-5-medium',
    'gpt-5-2025-08-07-high': 'gpt-5-high',
    'gpt-5-mini-2025-08-07-medium': 'gpt-5-mini-medium',
    'gpt-5-mini-2025-08-07-high': 'gpt-5-mini-high',
    'o3-2025-04-16-high': 'o3-high',
    'claude-sonnet-4-20250514': 'claude-sonnet-4',
    'claude-sonnet-4-5@20250929': 'claude-sonnet-4.5',
    'claude-opus-4-1-20250805': 'claude-opus-4.1',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'grok-4': 'grok-4',
    'grok-4-fast-reasoning': 'grok-4-fast-reasoning',
    'grok-code-fast-1': 'grok-code-fast-1',
}
NAME_TO_API = {name: api for api, name in API_TO_NAME.items()}

TRACE_ACCESSIBLE = list(API_TO_NAME.values())[:14]

# Models with enough fittable turns for the cognitive models (Llama-3.3-70B has < 20).
FITTED_MODELS = [m for m in TRACE_ACCESSIBLE if m != 'llama-v3.3-70b-instruct']
