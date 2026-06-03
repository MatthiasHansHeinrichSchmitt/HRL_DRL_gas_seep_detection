import importlib.util
import inspect

def load_reward_function_source(agent_module_path: str, reward_class_name: str = "Reward"):
    if not agent_module_path or not isinstance(agent_module_path, str):
        return ""
    spec = importlib.util.spec_from_file_location("agent_module", agent_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RewardClass = getattr(module, reward_class_name)
    reward_instance = RewardClass()
    return inspect.getsource(reward_instance.get_reward)