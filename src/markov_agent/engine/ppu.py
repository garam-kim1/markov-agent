from typing import Any, TypeVar

from pydantic import BaseModel

from markov_agent.core.state import BaseState
from markov_agent.engine.adk_wrapper import ADKConfig, ADKController, RetryPolicy
from markov_agent.engine.sampler import execute_parallel_sampling
from markov_agent.topology.node import BaseNode

StateT = TypeVar("StateT", bound=BaseState)


class ProbabilisticNode(BaseNode[StateT]):
    """
    A node that uses a Probabilistic Processing Unit (LLM via ADK)
    to determine the next state.
    """

    def __init__(
        self,
        name: str,
        adk_config: ADKConfig,
        prompt_template: str,
        output_schema: type[BaseModel] | None = None,
        samples: int = 1,
        retry_policy: RetryPolicy = None,
        mock_responder=None,
        state_updater=None,
        state_type: type[StateT] | None = None,
    ):
        super().__init__(name, state_type=state_type)
        self.adk_config = adk_config
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.samples = samples
        self.retry_policy = retry_policy or RetryPolicy()
        self.state_updater = state_updater

        # Inject native JSON support if schema is provided
        if self.output_schema:
            if self.adk_config.generation_config is None:
                self.adk_config.generation_config = {}

            # Configure native structured output
            self.adk_config.generation_config["response_mime_type"] = "application/json"
            # google-adk/genai often accepts the pydantic class or its schema
            self.adk_config.generation_config["response_schema"] = self.output_schema

        self.controller = ADKController(
            self.adk_config,
            self.retry_policy,
            mock_responder=mock_responder,
            output_schema=self.output_schema,
        )

    async def _run_async_impl(self, context: Any) -> Any:
        """
        Executes the PPU logic within the ADK runtime.
        """
        from google.adk.events import Event, EventActions
        
        # 1. Access State (Dict or Typed)
        # context.session.state is a dict
        state_dict = context.session.state
        
        # We try to use the typed state for prompt rendering if available
        state_obj = state_dict
        if self.state_type:
            try:
                state_obj = self.state_type.model_validate(state_dict)
            except Exception:
                # Fallback to construct if validation strictly fails (e.g. extra fields)
                # or just use the dict if we can't build the object.
                try:
                    state_obj = self.state_type.construct(**state_dict)
                except Exception:
                    pass

        # 2. Render Prompt
        # We need to render the prompt
        # We prefer passing the object itself to format, but str.format(**obj) doesn't work directly on objects.
        # We need to pass a dict that includes methods if possible, or just the dump.
        # However, users might use {get_context()} which implies the state object is available in the context?
        # Standard python format: "{foo}".format(foo=obj.foo)
        # But if the template is "{get_context()}", format() won't call the method.
        # Jinja2 handles this, but standard format does not.
        # Based on the example, the user uses "{get_context()}" inside the prompt template text?
        # No, the example prompt has: "Current Context:\n{get_context()}"
        # This implies f-string style, but we are doing .format().
        # Wait, the example code uses prompt_template.format(**state.model_dump()) usually.
        # BUT model_dump() does NOT include methods.
        # So prompt_template="{get_context()}" will FAIL with .format(**dump).
        # We need to inspect the template or just pre-calculate methods?
        # Or, we allow the prompt_template to be a callable?
        # NO, the example uses .format style syntax but assumes methods are available?
        # Actually, standard str.format can't call methods like that unless passed as kwargs.
        # e.g. .format(get_context=state.get_context())
        
        # Let's fix the Prompt Rendering to be smarter:
        # If state_obj is a model, we can try to evaluate expressions or pass methods as values.
        # For now, let's stick to the convention: pass model_dump() AND known methods?
        # Or simply support accessing attributes.
        # The specific error in the example suggests the prompt template is expecting {get_context()}?
        # The error I saw earlier was about `add_message` on dict.
        # Let's handle the prompt formatting by trying to bind methods if they are in the template keys?
        # Simplest fix for the specific example usage:
        # The example defines prompt_template with {get_context()}. 
        # We should calculate `get_context()` and pass it to format.
        
        render_kwargs = {}
        if isinstance(state_obj, BaseModel):
            render_kwargs.update(state_obj.model_dump())
            # Hack: inspect the state object for methods and add them to kwargs if they are 0-arg
            for attr_name in dir(state_obj):
                if not attr_name.startswith("_"):
                    attr = getattr(state_obj, attr_name)
                    if callable(attr):
                        try:
                            render_kwargs[attr_name] = attr() # Call it!
                        except Exception:
                            pass # Skip if it needs args
        elif isinstance(state_dict, dict):
            render_kwargs.update(state_dict)
            
        try:
            prompt = self.prompt_template.format(**render_kwargs)
        except Exception:
             # If formatting fails (e.g. missing key), we log or just use raw template
             # to avoid crashing, but usually we want to crash to alert the user.
             # We'll re-raise for visibility during dev
             raise

        # 3. Define generation task
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )
            
        # 4. Execute Parallel Sampling
        # This returns the best result (type depends on output_schema)
        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples
        )
        
        # 5. Update State
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()
            
        if self.state_updater:
             # Use state_updater. It usually expects (StateT, Result) -> StateT
             # If we have a typed state object, use it.
             if self.state_type and isinstance(state_obj, BaseModel):
                  updated_state = self.state_updater(state_obj, result)
                  if isinstance(updated_state, BaseModel):
                       context.session.state.update(updated_state.model_dump())
                  elif isinstance(updated_state, dict):
                       context.session.state.update(updated_state)
             else:
                  # Fallback for dict
                  updated_state = self.state_updater(state_dict, result)
                  if isinstance(updated_state, dict):
                       context.session.state.update(updated_state)
                  elif isinstance(updated_state, BaseModel):
                       context.session.state.update(updated_state.model_dump())
        else:
             # Try parse_result (for subclasses overriding it)
             # We only do this if we can form a valid state object, or if parse_result handles dicts?
             # Standard parse_result in ProbabilisticNode expects StateT.
             
             used_parse_result = False
             if self.state_type and isinstance(state_obj, BaseModel):
                  try:
                      updated_state = self.parse_result(state_obj, result)
                      if isinstance(updated_state, BaseModel):
                           context.session.state.update(updated_state.model_dump())
                           used_parse_result = True
                  except Exception:
                      # If parse_result fails (e.g. not implemented or type mismatch), fall through
                      pass
             
             if not used_parse_result:
                 # Default: append to history if exists
                 if "history" not in context.session.state:
                     context.session.state["history"] = []
                 context.session.state["history"].append({"node": self.name, "output": output_payload})
                 
                 # Also merge the result into state if it's a dict/model
                 if isinstance(output_payload, dict):
                     context.session.state.update(output_payload)

        # 6. Emit Event
        yield Event(
            author=self.name,
            actions=EventActions(), 
            # payload={"output": output_payload}  # ADK Event does not support payload
        )

    async def execute(self, state: StateT) -> StateT:
        """
        Legacy/Convenience wrapper. 
        Warning: This bypasses the ADK Runner mechanics and runs logic directly on the State object.
        """
        # Mimic the logic for standalone usage
        prompt = self._render_prompt(state)
        
        async def generate_task():
            return await self.controller.generate(
                prompt, output_schema=self.output_schema
            )

        result = await execute_parallel_sampling(
            generate_func=generate_task, k=self.samples
        )
        
        return self.parse_result(state, result)


    def _render_prompt(self, state: StateT) -> str:
        # Simple format
        try:
            return self.prompt_template.format(**state.model_dump())
        except Exception:
            # Fallback if state format fails
            return self.prompt_template

    def parse_result(self, state: StateT, result: Any) -> StateT:
        """
        Parses result. If state_updater is provided, uses it.
        Otherwise, default parser appends result to history.
        """
        # If a custom updater is provided, use it
        if self.state_updater:
            # The updater should return a NEW state instance (immutability)
            return self.state_updater(state, result)

        # Default behavior
        output_payload = result
        if isinstance(result, BaseModel):
            output_payload = result.model_dump()

        state.record_step({"node": self.name, "output": output_payload})
        return state
