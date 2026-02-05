from jinja2 import BaseLoader, Environment


class PromptEngine:
    """Jinja2-based structured prompting."""

    def __init__(self):
        self.env = Environment(loader=BaseLoader())

    def render(self, template_str: str, **kwargs) -> str:
        template = self.env.from_string(template_str)
        return template.render(**kwargs)
