import asyncio
import os
import random
from typing import Literal

from pydantic import BaseModel, Field

from markov_agent import ADKConfig, BaseState, Graph


# 1. State
class GameState(BaseState):
    health: int = 100
    gold: int = 10
    location: str = "Town"
    inventory: list[str] = Field(default_factory=list)
    turn: int = 0


# 2. Config
API_KEY = os.environ.get("GEMINI_API_KEY")
model = "gemini-3-flash-preview" if API_KEY else "openai/Qwen3-0.6B-Q4_K_M.gguf"

adk_config = ADKConfig(
    model_name=model,
    api_key=API_KEY or "no-key",
    use_litellm=not bool(API_KEY),
    temperature=0.7,
)


# 3. Nodes
class ActionOutput(BaseModel):
    action: Literal["Explore", "Rest", "Shop", "Boss"]
    thought: str


def update_state(state: GameState, result: ActionOutput) -> GameState:
    s = state.model_copy(deep=True)
    s.turn += 1

    log = f"Action: {result.action} ({result.thought})"

    if result.action == "Explore":
        found_gold = random.randint(10, 50)
        damage = random.randint(0, 20)
        s.gold += found_gold
        s.health -= damage
        s.location = "Wilderness"
        log += f" | Found {found_gold}g, took {damage} dmg."
    elif result.action == "Rest":
        heal = 30
        cost = 5
        if s.gold >= cost:
            s.health = min(100, s.health + heal)
            s.gold -= cost
            s.location = "Inn"
            log += f" | Healed {heal} for {cost}g."
        else:
            log += " | Not enough gold to rest!"
    elif result.action == "Shop":
        cost = 50
        if s.gold >= cost:
            s.gold -= cost
            s.inventory.append("Potion")
            s.location = "Shop"
            log += " | Bought Potion."
        else:
            log += " | Too poor for shop."
    elif result.action == "Boss":
        s.location = "Boss Lair"
        log += " | FIGHTING BOSS!"

    # Append log to history or just print
    # BaseState has 'history' but we can also use a custom log field if we want
    # For dashboard, let's just use the event log which DashboardRunner captures automatically
    # from the node output. But here we modify state.

    # Let's add a meta log for the dashboard to pick up
    if "logs" not in s.meta:
        s.meta["logs"] = []
    s.meta["logs"].append(log)

    return s


graph = Graph("RPG_Agent", state_type=GameState, max_steps=15)


@graph.node(adk_config=adk_config, state_updater=update_state)
def decide_action(state: GameState) -> ActionOutput:
    """
    RPG ADVENTURE - Turn {{turn}}
    Health: {{health}} | Gold: {{gold}} | Loc: {{location}}
    Inventory: {{inventory}}

    Choose next action carefully to survive and get rich.
    - Explore: Risk damage, find gold.
    - Rest: Heal (costs 5g).
    - Shop: Buy Potion (costs 50g).
    - Boss: ONLY if you have Potion and Health > 80.
    """
    raise NotImplementedError


# 4. Topology
# Simple loop
graph.add_transition("decide_action", "decide_action")


# 5. Run
async def main():
    # Only run if rich is installed (it is)
    print("Starting Dashboard Demo...")
    final_state = await graph.run_with_dashboard(GameState(), refresh_rate=4, delay=1.0)
    print(f"Simulation Ended. Final Gold: {final_state.gold}")


if __name__ == "__main__":
    asyncio.run(main())
