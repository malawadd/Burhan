"""Server-owned inspection capabilities; client model names never become paths or tools."""
import json

RESEARCH_MODELS = frozenset({'research-phele', 'research-fasteners', 'research-stain'})


def load_tasks(settings):
    tasks = json.loads(settings.catalogue.read_text(encoding='utf-8'))['tasks']
    expanded = (json.loads(settings.research_catalogue.read_text(encoding='utf-8'))['tasks']
                if settings.research_catalogue.exists() else [])
    for task in expanded:
        if (not task['id'].startswith('research:') or task.get('researchOnly') is not True
                or task['mode'] not in ('research', 'manual')):
            raise ValueError('Invalid expanded inspection task')
        if task['mode'] == 'research':
            if task.get('modelId') not in RESEARCH_MODELS or task.get('automatic') is not True:
                raise ValueError('Expanded detector must bind a registered bundled model')
        elif task.get('automatic') or task.get('modelId'):
            raise ValueError('Manual expanded checks cannot bind detectors')
    result = tasks + expanded
    if len({t['id'] for t in result}) != len(result):
        raise ValueError('Duplicate inspection capability identifier')
    return result


def available(task, context):
    if not task.get('researchOnly'):
        return True
    if context.get('expandedInspectionModels') is not True:
        return False
    model = task.get('modelId')
    return not model or (model in RESEARCH_MODELS and model in context.get('installedModelIds', []))


def capability_key(context):
    """Only registered IDs affect capability transitions, not arbitrary client labels."""
    return (context.get('expandedInspectionModels') is True,
            tuple(sorted(RESEARCH_MODELS.intersection(context.get('installedModelIds', [])))))


def selection_key(context):
    return (context.get('selectionMode', 'legacy'), context.get('activeTaskId', ''),
            context.get('selectionConfirmed', False), context.get('mode', 'manual'))
