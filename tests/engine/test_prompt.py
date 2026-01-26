from markov_agent.engine.prompt import PromptEngine


def test_prompt_engine_render():
    engine = PromptEngine()

    # Test simple variable substitution
    template = "Hello {{ name }}!"
    result = engine.render(template, name="World")
    assert result == "Hello World!"

    # Test logic (Jinja2 features)
    template_logic = "{% if show %}Visible{% else %}Hidden{% endif %}"
    assert engine.render(template_logic, show=True) == "Visible"
    assert engine.render(template_logic, show=False) == "Hidden"

    # Test list iteration
    template_list = "{% for item in items %}{{ item }}-{% endfor %}"
    assert engine.render(template_list, items=[1, 2, 3]) == "1-2-3-"
