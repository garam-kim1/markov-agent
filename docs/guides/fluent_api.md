# Fluent API & Decorators

The `markov-agent` library provides a high-level, fluent API designed for rapid prototyping and readable graph assembly. This "Pythonic" layer wraps the underlying Markovian structures.

## 1. Graph Decorators

The most efficient way to build agents is using the `@graph.node` and `@graph.task` decorators.

### @graph.node (Probabilistic Nodes)
Use this for nodes that require an LLM (PPU).

```python
g = Graph("MyAgent", state_type=MyState)

@g.node(samples=3, sampling_strategy="uniform")
async def draft_email(state: MyState):
    """
    You are an expert copywriter. 
    Task: Write an email about {{ product }}.
    """
```
*   **Prompt:** Taken from the function's docstring (dedented).
*   **State Type:** Automatically inferred from the first argument's type hint.
*   **Output Schema:** Inferred from the return type hint (if it's a Pydantic model).

### @graph.task (Functional Nodes)
Use this for deterministic Python logic (calculators, tools, lints).

```python
@g.task
def validate_output(state: MyState):
    return {"is_valid": len(state.draft) > 100}
```

## 2. Fluent Connections (The >> Operator)

Instead of manually calling `add_transition`, you can chain nodes using the `>>` (right shift) operator.

### Simple Chaining
```python
g.connect(draft_email >> validate_output >> send_email)
```

### Chaining with Logic
You can combine flows with more complex routing:
```python
g.connect(start_node >> next_node)
g.route("next_node", {
    "success_node": lambda s: s.score > 0.8,
    "retry_node": None  # Default
})
```

## 3. Visualization

Understanding complex topologies is easier with visualization.

```python
g.visualize()
```
This prints a **Mermaid.js** flowchart directly to your console using `rich`.

## 4. Integrated Lifecycle

The `Graph` object acts as a control center for the entire development lifecycle:

```python
# 1. Define
g = Graph(...)

# 2. Visualize
g.visualize()

# 3. Analyze
analyzer = g.analyze()
print(analyzer.generate_mermaid_graph(analyzer.extract_matrix()))

# 4. Simulate
results = await g.simulate(dataset=test_cases, n_runs=10)

# 5. Execute
final_state = await g(initial_state) # __call__ shortcut
```
