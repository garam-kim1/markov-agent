import asyncio
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from markov_agent.engine.adk_wrapper import ADKConfig
from markov_agent.engine.ppu import ProbabilisticNode
from markov_agent.core.state import BaseState

class SimpleState(BaseState):
    response: str | None = None

async def main():
    print("Initializing Local LLM Test...")
    
    # Configuration for the local model
    config = ADKConfig(
        model_name="openai/Qwen3-0.6B-Q4_K_M.gguf",
        api_base="http://192.168.1.213:8080/v1",
        api_key="dummy", 
        use_litellm=True,
        temperature=0.7
    )

    node = ProbabilisticNode(
        name="local_tester",
        adk_config=config,
        prompt_template="Say hello and tell me what model you are.",
        state_type=SimpleState
    )

    state = SimpleState()
    print("Executing Node...")
    
    try:
        # execute returns the updated state
        result_state = await node.execute(state)
        
        print("\nExecution Successful!")
        print("Last History Entry:")
        print(result_state.history[-1])
        
    except Exception as e:
        print(f"\nExecution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
