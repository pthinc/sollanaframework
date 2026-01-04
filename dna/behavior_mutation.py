# behavior_mutation.py
import random
import math
import time
import copy
import torch
import torch.nn.functional as F
from typing import Callable, Dict, Any, List, Tuple, Optional

Seed = Optional[int]

class BehaviorVariant:
    def __init__(self, base_id: str, func: Callable[[float], float], metadata: Dict[str, Any]):
        self.base_id = base_id
        self.func = func
        self.metadata = metadata
        self.created_at = time.time()
        self.score = None

class BehaviorMutator:
    def __init__(self, variation_range: float = 0.05, seed: Seed = None):
        self.variation_range = variation_range
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def mutate_scalar_function(self, base_fn: Callable[[float], float]) -> Callable[[float], float]:
        delta = random.uniform(-self.variation_range, self.variation_range)
        def mutated(x: float) -> float:
            return base_fn(x) + delta
        return mutated

    def mutate_vectorized_fn(self, base_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        noise_scale = self.variation_range
        def mutated(x: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(x) * noise_scale
            return base_fn(x) + noise
        return mutated

class MetaEvaluator:
    def __init__(self, novelty_weight: float = 1.0, safety_weight: float = 2.0, utility_weight: float = 1.0):
        self.novelty_weight = novelty_weight
        self.safety_weight = safety_weight
        self.utility_weight = utility_weight

    def novelty_score(self, variant: BehaviorVariant, archive: List[BehaviorVariant]) -> float:
        if not archive:
            return 1.0
        similarities = []
        for v in archive:
            sim = self._rough_similarity(variant, v)
            similarities.append(sim)
        mean_sim = sum(similarities) / len(similarities)
        return 1.0 - mean_sim

    def safety_score(self, variant: BehaviorVariant, context: Dict[str, Any]) -> float:
        unsafe_tags = context.get("unsafe_tags", set())
        tags = set(variant.metadata.get("tags", []))
        penalty = 1.0 if tags & unsafe_tags else 0.0
        return 1.0 - penalty

    def utility_score(self, variant: BehaviorVariant, task_signal: float) -> float:
        return max(0.0, min(1.0, task_signal))

    def evaluate(self, variant: BehaviorVariant, archive: List[BehaviorVariant], context: Dict[str, Any], task_signal: float) -> float:
        n = self.novelty_score(variant, archive)
        s = self.safety_score(variant, context)
        u = self.utility_score(variant, task_signal)
        score = self.novelty_weight * n + self.safety_weight * s + self.utility_weight * u
        variant.score = score
        return score

    def _rough_similarity(self, a: BehaviorVariant, b: BehaviorVariant) -> float:
        ma = set(a.metadata.get("tags", []))
        mb = set(b.metadata.get("tags", []))
        if not ma and not mb:
            return 0.0
        inter = len(ma & mb)
        union = len(ma | mb)
        return inter / union if union > 0 else 0.0

class RandomBehaviorGenerator:
    def __init__(self, mutator: BehaviorMutator, evaluator: MetaEvaluator, memory, dna_model, archive_limit: int = 1000):
        self.mutator = mutator
        self.evaluator = evaluator
        self.memory = memory
        self.dna_model = dna_model
        self.archive: List[BehaviorVariant] = []
        self.archive_limit = archive_limit

    def propose_from_base(self, base_id: str, base_fn: Callable, metadata: Dict[str, Any], context: Dict[str, Any], task_signal: float) -> Optional[BehaviorVariant]:
        if isinstance(base_fn, torch.nn.Module):
            mutated = self._mutate_module(base_fn)
            variant_fn = mutated
        elif callable(base_fn):
            try:
                variant_fn = self.mutator.mutate_scalar_function(base_fn)
            except Exception:
                variant_fn = self.mutator.mutate_vectorized_fn(base_fn)
        else:
            return None
        variant = BehaviorVariant(base_id=base_id, func=variant_fn, metadata=metadata)
        score = self.evaluator.evaluate(variant, self.archive, context, task_signal)
        if self._accept(score, context):
            self._archive(variant)
            self._commit_to_memory(variant, context)
            return variant
        return variant if score > 0 else None

    def _accept(self, score: float, context: Dict[str, Any]) -> bool:
        threshold = context.get("accept_threshold", 1.0)
        return score >= threshold

    def _archive(self, variant: BehaviorVariant):
        self.archive.append(variant)
        if len(self.archive) > self.archive_limit:
            self.archive.pop(0)

    def _commit_to_memory(self, variant: BehaviorVariant, context: Dict[str, Any]):
        trace_id = f"var_{int(time.time()*1000)}"
        delta_N = variant.score
        decay_rate = context.get("decay_rate", 0.01)
        self.memory.trigger_behavior(trace_id, context=context.get("context", "mutant"), delta_N=delta_N, decay_rate=decay_rate)
        variant.metadata["trace_id"] = trace_id

    def _mutate_module(self, module: torch.nn.Module) -> torch.nn.Module:
        mod = copy.deepcopy(module)
        for p in mod.parameters():
            noise = torch.randn_like(p) * self.mutator.variation_range
            p.data.add_(noise)
        return mod

class EvolutionaryController:
    def __init__(self, generator: RandomBehaviorGenerator, population_size: int = 20, mutation_rate: float = 0.2):
        self.generator = generator
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def run_generation(self, base_pool: List[Tuple[str, Callable, Dict[str, Any]]], context: Dict[str, Any], task_signal_fn: Callable[[Dict[str, Any]], float]) -> List[BehaviorVariant]:
        population: List[BehaviorVariant] = []
        for base_id, base_fn, meta in base_pool:
            task_signal = task_signal_fn(meta)
            variant = self.generator.propose_from_base(base_id, base_fn, meta, context, task_signal)
            if variant is not None:
                population.append(variant)
        population.sort(key=lambda v: v.score or 0.0, reverse=True)
        survivors = population[:self.population_size]
        self._breed_and_mutate(survivors, context, task_signal_fn)
        return survivors

    def _breed_and_mutate(self, survivors: List[BehaviorVariant], context: Dict[str, Any], task_signal_fn: Callable[[Dict[str, Any]], float]):
        for _ in range(int(self.population_size * self.mutation_rate)):
            parent = random.choice(survivors)
            base_fn = parent.func
            meta = parent.metadata
            new_variant = self.generator.propose_from_base(parent.base_id, base_fn, meta, context, task_signal_fn(meta))
            if new_variant:
                survivors.append(new_variant)
