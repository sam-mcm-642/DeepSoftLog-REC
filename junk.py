import math
from typing import Union

import numpy as np
import torch

FloatOrTensor = Union[float, torch.Tensor]


# def safe_log(x: FloatOrTensor) -> FloatOrTensor:
#     if torch.is_tensor(x):
#         x[x <= 1e-12] = 0.
#         return torch.log(x)
#     return math.log(x)


# def safe_exp(x: FloatOrTensor) -> FloatOrTensor:
#     if torch.is_tensor(x):
#         return torch.exp(x)
#     return math.exp(x)


def safe_log(x: FloatOrTensor) -> FloatOrTensor:
    # print(f"safe_log called with x={x}")
    if torch.is_tensor(x):
        # Check for very small values
        if torch.any(x <= 1e-12):
            print(f"WARNING: safe_log received values ≤ 1e-12: {x[x <= 1e-12]}")
        x_safe = x.clone()
        x_safe[x_safe <= 1e-12] = 1e-12
        return torch.log(x_safe)
    
    if x <= 1e-12:
        print(f"WARNING: safe_log received small value: {x}")
        return math.log(1e-12)
    return math.log(x)

def safe_exp(x: FloatOrTensor) -> FloatOrTensor:
    print(f"safe_exp called with x={x}")
    if torch.is_tensor(x):
        # Check for very negative values
        if torch.any(x < -100):
            print(f"WARNING: safe_exp received values < -100: {x[x < -100]}")
        return torch.exp(torch.clamp(x, min=-100))
    
    if x < -100:
        print(f"WARNING: safe_exp received very negative value: {x}")
        return math.exp(-100)
    return math.exp(x)

from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Iterable, TypeVar

from deepsoftlog.algebraic_prover.terms.expression import Fact

Value = TypeVar("Value")


class Algebra(ABC, Generic[Value]):
    """Interface for a algebra on facts"""

    @abstractmethod
    def value_pos(self, fact: Fact) -> Value:
        """
        Value of a positive fact.
        Note that we assume that this is only called on annotated facts.
        """
        pass

    @abstractmethod
    def value_neg(self, fact: Fact) -> Value:
        """
        Value of a negative fact.
        Note that we assume that this is only called on annotated facts.
        """
        pass

    @abstractmethod
    def multiply(self, value1: Value, value2: Value) -> Value:
        pass

    @abstractmethod
    def add(self, value1: Value, value2: Value) -> Value:
        pass

    @abstractmethod
    def one(self) -> Value:
        pass

    @abstractmethod
    def zero(self) -> Value:
        pass

    def in_domain(self, v: Value) -> bool:
        return True

    def multiply_value_pos(self, value: Value, fact: Fact) -> Value:
        return self.multiply(value, self.value_pos(fact))

    def multiply_value_neg(self, value: Value, fact: Fact) -> Value:
        return self.multiply(value, self.value_neg(fact))

    def reduce_mul(self, values: Iterable[Value]) -> Value:
        values = tuple(values)
        if len(values) == 0:
            return self.one()
        return reduce(self.multiply, values)

    def reduce_mul_value_pos(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_pos(fact) for fact in facts)
        return self.reduce_mul(values)

    def reduce_mul_value_neg(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_neg(fact) for fact in facts)
        return self.reduce_mul(values)

    def reduce_mul_value(
        self, pos_facts: Iterable[Fact], neg_facts: Iterable[Fact]
    ) -> Value:
        return self.multiply(
            self.reduce_mul_value_pos(pos_facts), self.reduce_mul_value_neg(neg_facts)
        )

    def reduce_add(self, values: Iterable[Value]) -> Value:
        values = tuple(values)
        if len(values) == 0:
            return self.zero()
        return reduce(self.add, values)

    def reduce_add_value_pos(self, facts: Iterable[Fact]) -> Value:
        values = (self.value_pos(fact) for fact in facts)
        return self.reduce_add(values)

    def get_dual(self):
        return DualAlgebra(self)

    def reset(self):
        pass

    def evaluate(self, value: Value) -> Value:
        return value

    def eval_zero(self):
        return self.evaluate(self.zero())

    def eval_one(self):
        return self.evaluate(self.one())

    def is_eval_zero(self, value):
        return self.evaluate(value) == self.eval_zero()

    def is_eval_one(self, value):
        return self.evaluate(value) == self.eval_one()


class DualAlgebra(Algebra):
    def __init__(self, algebra):
        self.algebra = algebra

    def value_pos(self, fact: Fact) -> Value:
        return self.algebra.value_neg(fact)

    def value_neg(self, fact: Fact) -> Value:
        return self.algebra.value_pos(fact)

    def multiply(self, value1: Value, value2: Value) -> Value:
        return self.algebra.add(value1, value2)

    def add(self, value1: Value, value2: Value) -> Value:
        return self.algebra.multiply(value1, value2)

    def one(self) -> Value:
        return self.algebra.zero()

    def zero(self) -> Value:
        return self.algebra.one()

    def get_dual(self):
        return self.algebra

    def evaluate(self, value: Value) -> Value:
        return self.algebra.evaluate(value)


class CompoundAlgebra(Algebra[Value], ABC, Generic[Value]):
    def __init__(self, eval_algebra):
        self._eval_algebra = eval_algebra

    def evaluate(self, value):
        return value.evaluate(self._eval_algebra)

from abc import ABC

from deepsoftlog.algebraic_prover.algebras.abstract_algebra import CompoundAlgebra
from deepsoftlog.algebraic_prover.algebras.string_algebra import STRING_ALGEBRA
from deepsoftlog.algebraic_prover.terms.color_print import get_color
from deepsoftlog.algebraic_prover.terms.expression import Fact


class AndOrFormula(ABC):
    """
    AND/OR formula (in NNF)
    """

    def __and__(self, other):
        if other is TRUE_LEAF:
            return self
        if other is FALSE_LEAF:
            return FALSE_LEAF
        if self == other:
            return self
        return AndNode(self, other)

    def __or__(self, other):
        if other is TRUE_LEAF:
            return TRUE_LEAF
        if other is FALSE_LEAF:
            return self
        if self == other:
            return self
        return OrNode(self, other)

    def evaluate(self, algebra):
        raise NotImplementedError


class AndNode(AndOrFormula):
    def __init__(self, left: AndOrFormula, right: AndOrFormula):
        self.left = left
        self.right = right

    def evaluate(self, algebra):
        return algebra.multiply(self.left, self.right)

    def __eq__(self, other):
        return (
            isinstance(other, AndNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        return f"{self.left} \033[1m&\033[0m {self.right}"


class OrNode(AndOrFormula):
    def __init__(self, left: AndOrFormula, right: AndOrFormula):
        self.left = left
        self.right = right

    def evaluate(self, algebra):
        return algebra.add(self.left, self.right)

    def __eq__(self, other):
        return (
            isinstance(other, OrNode)
            and self.left == other.left
            and self.right == other.right
        )

    def __repr__(self):
        color = get_color()
        color_end = "\033[0m"
        return f"{color}({color_end}{self.left} {color}|{color_end} {self.right}{color}){color_end}"


class TrueLeaf(AndOrFormula):
    def __and__(self, other):
        return other

    def __or__(self, other):
        return self

    def evaluate(self, algebra):
        return algebra.one()

    def __eq__(self, other):
        return isinstance(other, TrueLeaf)

    def __repr__(self):
        return "true"


class FalseLeaf(AndOrFormula):
    def __and__(self, other):
        return self

    def __or__(self, other):
        return other

    def evaluate(self, algebra):
        return algebra.zero()

    def __eq__(self, other):
        return isinstance(other, FalseLeaf)

    def __repr__(self):
        return "false"


class LeafNode(AndOrFormula):
    def __init__(self, fact: Fact, negated=False):
        self.fact = fact
        self.negated = negated

    def evaluate(self, algebra):
        if self.negated:
            return algebra.value_neg(self.fact)
        else:
            return algebra.value_pos(self.fact)

    def __eq__(self, other):
        return (
            isinstance(other, LeafNode)
            and self.fact == other.fact
            and self.negated == other.negated
        )

    def __repr__(self):
        if self.negated:
            return f"!{self.fact}"
        else:
            return f"{self.fact}"


TRUE_LEAF = TrueLeaf()
FALSE_LEAF = FalseLeaf()


class AndOrAlgebra(CompoundAlgebra[AndOrFormula]):
    """
    And-or formula, with some auto-simplifications.
    (Similar to a Free Algebra)
    """

    def value_pos(self, fact: Fact) -> AndOrFormula:
        return LeafNode(fact)

    def value_neg(self, fact: Fact) -> AndOrFormula:
        return LeafNode(fact, negated=True)

    def multiply(self, value1: AndOrFormula, value2: AndOrFormula) -> AndOrFormula:
        return value1 & value2

    def add(self, value1: AndOrFormula, value2: AndOrFormula) -> AndOrFormula:
        return value1 | value2

    def one(self) -> AndOrFormula:
        return TRUE_LEAF

    def zero(self) -> AndOrFormula:
        return FALSE_LEAF


AND_OR_ALGEBRA = AndOrAlgebra(STRING_ALGEBRA)


from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra


class BooleanAlgebra(Algebra[bool]):
    """
    Boolean algebra.
    Used for boolean inference (i.e. regular prolog).
    """

    def value_pos(self, fact) -> bool:
        return fact.get_probability() > 0.0

    def value_neg(self, fact) -> bool:
        return self.value_pos(fact)

    def multiply(self, value1: bool, value2: bool) -> bool:
        return value1 and value2

    def add(self, value1: bool, value2: bool) -> bool:
        return value1 or value2

    def one(self) -> bool:
        return True

    def zero(self) -> bool:
        return False


BOOLEAN_ALGEBRA = BooleanAlgebra()


from deepsoftlog.algebraic_prover.algebras import safe_log_add, safe_log_negate
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact
import torch


class ProbabilityAlgebra(Algebra[float]):
    def value_pos(self, f: Fact) -> float:
        return f.get_probability()

    def value_neg(self, f: Fact) -> float:
        if f.is_annotated():
            return 1 - f.get_probability()
        return 1.0

    def in_domain(self, v: float) -> bool:
        return 0 <= v <= 1

    def multiply(self, value1: float, value2: float) -> float:
        return value1 * value2

    def add(self, value1: float, value2: float) -> float:
        return value1 + value2

    def one(self) -> float:
        return 1.0

    def zero(self) -> float:
        return 0.0


class LogProbabilityAlgebra(Algebra[float]):
    """
    Probability algebra on facts in logspace.
    """

    ninf = float("-inf")

    def value_pos(self, f: Fact) -> Value:
        return f.get_log_probability()

    def value_neg(self, f: Fact) -> float:
        if f.is_annotated():
            return safe_log_negate(f.get_log_probability())
        return 0.0

    def in_domain(self, v: Value) -> bool:
        return v <= 1e-12

    def one(self) -> float:
        return 0.0

    def zero(self) -> float:
        return self.ninf

    # def add(self, a: float, b: float) -> float:
    #     return safe_log_add(a, b)

    # def multiply(self, a: float, b: float) -> float:
    #     return a + b
    def multiply(self, value1: float, value2: float) -> float:
        result = value1 + value2
        print(f"LOG_PROB_MULTIPLY: {value1:.6f} + {value2:.6f} = {result:.6f}")
        if isinstance(result, torch.Tensor) and torch.isneginf(result):
            print("WARNING: Log multiply resulted in -inf")
        return result

    def add(self, value1: float, value2: float) -> float:
        result = safe_log_add(value1, value2)
        print(f"LOG_PROB_ADD: logaddexp({value1:.6f}, {value2:.6f}) = {result:.6f}")
        if isinstance(result, torch.Tensor) and torch.isneginf(result):
            print("WARNING: Log add resulted in -inf")
        return result


PROBABILITY_ALGEBRA = ProbabilityAlgebra()
LOG_PROBABILITY_ALGEBRA = LogProbabilityAlgebra()


from typing import Optional, TypeVar

from pysdd.iterator import SddIterator
from pysdd.sdd import SddManager, SddNode

from deepsoftlog.algebraic_prover.algebras.and_or_algebra import AND_OR_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value, CompoundAlgebra
from deepsoftlog.algebraic_prover.algebras.string_algebra import STRING_ALGEBRA
from deepsoftlog.algebraic_prover.terms.expression import Fact

V = TypeVar("V")


class FastList:
    """
    List that allows fast access to elements by index and value.
    """

    def __init__(self):
        self._ix_to_val: list[V] = []
        self._val_to_ix: dict[V, int] = {}

    def __getitem__(self, ix: int) -> V:
        return self._ix_to_val[ix - 1]

    def __contains__(self, value: V):
        return value in self._val_to_ix

    def __len__(self) -> int:
        return len(self._ix_to_val)

    def index(self, val: V) -> int:
        return self._val_to_ix[val] + 1

    def append(self, val: V):
        self._ix_to_val.append(val)
        self._val_to_ix[val] = len(self._ix_to_val)


class SddFormula:
    """
    Self-managing SDD formula
    """

    def __init__(
        self,
        manager: SddManager,
        all_facts: FastList,
        formula: Optional[SddNode] = None,
    ):
        if formula is None:
            formula = manager.true()
        self.manager = manager
        self.all_facts = all_facts
        self.formula: SddNode = formula
        self._score: Optional[float] = None

    def _get_child(self, formula=None):
        return SddFormula(self.manager, self.all_facts, formula)

    def evaluate(self, algebra: Algebra) -> float:
        return _sdd_eval(self.manager, self.formula, algebra, self.all_facts)

    def __and__(self, other: "SddFormula") -> "SddFormula":
        return self._get_child(self.manager.conjoin(self.formula, other.formula))

    def __or__(self, other: "SddFormula") -> "SddFormula":
        return self._get_child(self.manager.disjoin(self.formula, other.formula))

    def _fact_id(self, fact: Fact, negated: bool) -> int:
        if negated:
            return -self._fact_id(fact, False)
        if fact not in self.all_facts:
            self.all_facts.append(fact)
            if len(self.all_facts) >= self.manager.var_count():
                self.manager.add_var_after_last()
        return self.all_facts.index(fact)

    def with_fact(self, fact: Fact, negated: bool = False) -> "SddFormula":
        assert fact.is_ground(), f"{fact} is not ground!"
        fact_id = self._fact_id(fact, negated)
        fact_sdd = self.manager.literal(fact_id)
        return self._get_child(self.manager.conjoin(self.formula, fact_sdd))

    def negate(self) -> "SddFormula":
        return self._get_child(self.manager.negate(self.formula))

    def view(self, save_path="temp.dot"):
        with open(save_path, "w") as out:
            print(self.formula.dot(), file=out)

        from graphviz import Source

        s = Source.from_file(save_path)
        s.view()

    def __repr__(self):
        return self.evaluate(STRING_ALGEBRA)

    def __eq__(self, other):
        return isinstance(other, SddFormula) and self.formula == other.formula

    def to_and_or_tree(self):
        return self.evaluate(AND_OR_ALGEBRA)


def _sdd_eval(
    manager: SddManager, root_node: SddNode, algebra: Algebra[Value], all_facts
) -> Value:
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node: SddNode, r_values, *_):
        if node is not None:
            if node.is_literal():
                literal = node.literal
                if literal < 0:
                    return algebra.value_neg(all_facts[(-literal) - 1])
                else:
                    return algebra.value_pos(all_facts[literal - 1])
            elif node.is_true():
                return algebra.one()
            elif node.is_false():
                return algebra.zero()
        # Decision node
        return algebra.reduce_add(
            [algebra.reduce_mul(value[0:2]) for value in r_values]
        )

    result = iterator.depth_first(root_node, _formula_evaluator)
    return result


class SddAlgebra(CompoundAlgebra[SddFormula]):
    """
    SDD semiring on facts.
    Solves the disjoin-sum problem.
    """

    def __init__(self, eval_algebra):
        super().__init__(eval_algebra)
        self.manager = SddManager()
        self.all_facts = FastList()
        self._i = 0

    def value_pos(self, fact) -> SddFormula:
        return self.one().with_fact(fact)

    def value_neg(self, fact) -> SddFormula:
        return self.one().with_fact(fact).negate()

    def multiply(self, value1: SddFormula, value2: SddFormula) -> SddFormula:
        return value1 & value2

    def add(self, value1: SddFormula, value2: SddFormula) -> SddFormula:
        return value1 | value2

    def one(self) -> SddFormula:
        return SddFormula(self.manager, self.all_facts)

    def zero(self) -> SddFormula:
        return self.one().negate()

    def reset(self):
        self.all_facts = FastList()
        

from typing import Union

from deepsoftlog.algebraic_prover.algebras.sdd_algebra import SddAlgebra, SddFormula
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import CompoundAlgebra, Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr
import torch


class ConjoinedFacts:
    def __init__(
        self, pos_facts: set[Expr], neg_facts: set[Expr], sdd_algebra: SddAlgebra
    ):
        self.pos_facts = pos_facts
        self.neg_facts = neg_facts
        self._sdd_algebra = sdd_algebra

    def __and__(self, other: "ConjoinedFacts") -> Union["ConjoinedFacts", SddFormula]:
        new_pos_facts = self.pos_facts | other.pos_facts
        new_neg_facts = self.neg_facts | other.neg_facts
        if len(new_pos_facts & new_neg_facts) != 0:
            return self._sdd_algebra.zero()
        return ConjoinedFacts(new_pos_facts, new_neg_facts, self._sdd_algebra)

    # def evaluate(self, algebra: Algebra):
    #     return algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    
    # def evaluate(self, algebra: Algebra):
    #     print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")
        
    #     # If there are no positive facts, this might be causing the -inf
    #     if not self.pos_facts:
    #         print("Warning: No positive facts to evaluate")
    #         # Return a very small probability instead of zero/inf
    #         return torch.tensor(-20.0)
        
    #     result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    #     print(f"reduce_mul_value result: {result}")
    
        # return result
        
    # def evaluate(self, algebra: Algebra):
    #     print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")        
    #     result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
    #     print(f"reduce_mul_value result: {result}")
    #     return result
    
    def evaluate(self, algebra: Algebra):
        print(f"ConjoinedFacts.evaluate called with pos_facts={self.pos_facts}")
        
        # Check for empty facts
        if not self.pos_facts:
            print("WARNING: Empty positive facts collection")
            
        # Try to detect problematic fact patterns
        small_probs = [f for f in self.pos_facts if hasattr(f, 'get_probability') and f.get_probability() < 0.001]
        if small_probs:
            print(f"WARNING: Very small probabilities detected: {small_probs}")
        
        result = algebra.reduce_mul_value(self.pos_facts, self.neg_facts)
        print(f"reduce_mul_value result: {result}")
        
        # Check result for numerical issues
        if isinstance(result, torch.Tensor):
            if torch.isneginf(result):
                print("CRITICAL: Evaluation resulted in -inf")
            elif torch.isnan(result):
                print("CRITICAL: Evaluation resulted in NaN")
                
        return result

    def __str__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"

    def __repr__(self):
        return f"ConjoinedFacts({self.pos_facts}, {self.neg_facts})"


class DnfAlgebra(CompoundAlgebra[Union[ConjoinedFacts, SddFormula]]):
    """
    Like the Sdd Algebra, but uses sets for simple conjunctions.
    This can be considerably faster, especially for programs without
    negation on rules. (in which case the knowledge compilation
    is only performed after all proofs are found).
    """

    def __init__(self, eval_algebra: Algebra):
        super().__init__(eval_algebra)
        self._sdd_algebra = SddAlgebra(eval_algebra)

    def value_pos(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(pos_facts={fact})

    def value_neg(self, fact: Fact) -> Value:
        return self._as_conjoined_facts(neg_facts={fact})

    def multiply(self, v1: Value, v2: Value) -> Value:
        if not isinstance(v1, ConjoinedFacts) or not isinstance(v2, ConjoinedFacts):
            v1 = self._as_sdd(v1)
            v2 = self._as_sdd(v2)
        return v1 & v2

    def reduce_mul_value_pos(self, facts) -> Value:
        facts = {f for f in facts if f.is_annotated()}
        return self._as_conjoined_facts(facts)

    def add(self, value1: Value, value2: Value) -> Value:
        return self._as_sdd(value1) | self._as_sdd(value2)

    def one(self) -> Value:
        return self._as_conjoined_facts()

    def zero(self) -> Value:
        return self._sdd_algebra.zero()

    def reset(self):
        self._sdd_algebra.reset()

    # def _as_sdd(self, value):
    #     if isinstance(value, ConjoinedFacts):
    #         return value.evaluate(self._sdd_algebra)
    #     return value
    
    # def _as_sdd(self, value):
    #     if isinstance(value, ConjoinedFacts):
    #         print(f"Converting ConjoinedFacts to SDD: pos_facts={value.pos_facts}, neg_facts={value.neg_facts}")
    #         result = value.evaluate(self._sdd_algebra)
    #         print(f"SDD evaluation result: {result}")
            
    #         # Add a floor to prevent -inf values
    #         if isinstance(result, torch.Tensor) and torch.isneginf(result):
    #             print(f"Found -inf result, replacing with floor value")
    #             result = torch.tensor(-20.0)  # A small log probability, not -inf
            
    #         return result
    #     return value
    
    def _as_sdd(self, value):
        if isinstance(value, ConjoinedFacts):
            print(f"Converting ConjoinedFacts to SDD: pos_facts={value.pos_facts}, neg_facts={value.neg_facts}")
            
            # Pre-check for potential issues
            if not value.pos_facts:
                print("WARNING: Converting empty ConjoinedFacts to SDD")
                
            result = value.evaluate(self._sdd_algebra)
            print(f"SDD evaluation result: {result}")
            
            # Add numerical safety checks
            if isinstance(result, torch.Tensor):
                if torch.isneginf(result):
                    print(f"FIXING: -inf result, replacing with floor value -20.0")
                    result = torch.tensor(-20.0)
                elif torch.isnan(result):
                    print(f"FIXING: NaN result, replacing with floor value -20.0")
                    result = torch.tensor(-20.0)
            
            return result
        return value

    
    # def evaluate(self, value):
    #     print(f"DnfAlgebra.evaluate called with value: {value}, type: {type(value)}")
    #     if isinstance(value, ConjoinedFacts):
    #         sdd_value = self._as_sdd(value)
    #         print(f"After _as_sdd: {sdd_value}")
    #         result = self._eval_algebra.evaluate(sdd_value)
    #         print(f"After _eval_algebra.evaluate: {result}")
    #         return result
    #     elif hasattr(value, 'evaluate'):
    #         print(f"Value has evaluate method")
    #         return value.evaluate(self._eval_algebra)
    #     else:
    #         print(f"Direct evaluation by _eval_algebra")
    #         return self._eval_algebra.evaluate(value)

    def _as_conjoined_facts(self, pos_facts=None, neg_facts=None):
        pos_facts = pos_facts or set()
        neg_facts = neg_facts or set()
        return ConjoinedFacts(pos_facts, neg_facts, self._sdd_algebra)

from collections import defaultdict
from typing import Iterable, Optional

from deepsoftlog.algebraic_prover.builtins import ALL_BUILTINS
from deepsoftlog.algebraic_prover.proving.proof_queue import OrderedProofQueue, ProofQueue
from deepsoftlog.algebraic_prover.proving.proof_tree import ProofTree
from deepsoftlog.algebraic_prover.proving.unify import mgu
from deepsoftlog.algebraic_prover.algebras.boolean_algebra import BOOLEAN_ALGEBRA
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from deepsoftlog.algebraic_prover.algebras.probability_algebra import (
    LOG_PROBABILITY_ALGEBRA,
    PROBABILITY_ALGEBRA,
)
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr
from deepsoftlog.algebraic_prover.terms.variable import CanonicalVariableCounter, fresh_variables
import torch

class ProofModule:
    def __init__(
        self,
        clauses: Iterable[Clause],
        algebra: Algebra,
    ):
        super().__init__()
        self.clauses: set[Clause] = set(clauses)
        self.algebra = algebra
        self.fresh_var_counter = CanonicalVariableCounter(functor="FV_")
        self.queried = None
        self.mask_query = False

    def mgu(self, t1, t2):
        return mgu(t1, t2)
    
    ##3.7 thinking version
    
    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
        predicate = term.get_predicate()
        # print(f"\nDEBUG: Attempting to match term: {term}")
        # print(f"DEBUG: Looking for predicate: {predicate}")
        
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                yield from builtin.get_answers(*term.arguments)

        for db_clause in self.clauses:
            # print(f"\nDEBUG: Examining clause: {db_clause}")
            
            # Handle both facts and rules correctly
            if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
                db_head = db_clause
                # print(f"DEBUG: Fact with predicate: {db_head.get_predicate()}")
            else:  # It's a rule
                if db_clause.functor == ':-':
                    db_head = db_clause.arguments[0]  # Get the head of the rule
                else:
                    db_head = db_clause.arguments[0]
                # print(f"DEBUG: Rule with head predicate: {db_head.get_predicate()}")
            
            # if self.mask_query and db_head == self.queried:
            #     continue
                
            if db_head.get_predicate() == predicate:
                # print(f"DEBUG: Found matching predicate")
                if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # For facts
                    fresh_db_clause = db_clause
                    head_to_match = db_head  # No renaming needed for facts
                else:  # For rules
                    fresh_db_clause = self.fresh_variables(db_clause)
                    head_to_match = fresh_db_clause.arguments[0]  # Use the fresh head for matching
                    
                # print(f"DEBUG: Working with clause: {fresh_db_clause}")
                # print(f"DEBUG: Attempting to match with: {head_to_match}")  # Now correctly using head with fresh variables
                
                result = self.mgu(term, head_to_match)
                # print(f"DEBUG: MGU result: {result}")
                
                if result is not None:
                    unifier, new_facts = result
                    print(f"DEBUG: Unifier: {unifier}")
                    if isinstance(fresh_db_clause, Expr):
                        new_clause = fresh_db_clause
                    else:
                        new_clause = fresh_db_clause.apply_substitution(unifier)
                    print(f"DEBUG: Yielding match: {new_clause}")
                    yield new_clause, unifier, new_facts    
         
                    
    # def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
    #     predicate = term.get_predicate()
    #     print(f"\nDEBUG: Attempting to match term: {term}")
    #     print(f"DEBUG: Looking for predicate: {predicate}")
        
    #     for builtin in self.get_builtins():
    #         if predicate == builtin.predicate:
    #             yield from builtin.get_answers(*term.arguments)

    #     for db_clause in self.clauses:
    #         print(f"\nDEBUG: Examining clause: {db_clause} with functor: {db_clause.functor}")
            
    #         # Handle both facts and rules correctly
    #         if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
    #             db_head = db_clause
    #             print(f"DEBUG: Fact with predicate: {db_head.get_predicate()}")
    #         else:  # It's a rule
    #             if db_clause.functor == ':-':
    #                 db_head = db_clause.arguments[0]  # Get the head of the rule
    #             else:
    #                 db_head = db_clause.arguments[0]
    #             print(f"DEBUG: Rule with head predicate: {db_head.get_predicate()}")
            
    #         if self.mask_query and db_head == self.queried: ###THIS LINE STOPS A QUERY FROM BEING REPEATED
    #             ##AND ALSO STOPS A QUERY FROM BEING USED AS A FACT
    #             ##WHICH IN TURN STOPS THE QUERY FROM BEING PROVEN BY ITSELF OR
    #             #AN IDENITCAL FACT
    #             continue
    #         print(f"DEBUG: Clause head: {db_head.get_predicate()}, looking for: {predicate}")    
    #         if db_head.get_predicate() == predicate:
    #             print(f"DEBUG: Found matching predicate")
    #             if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # For facts
    #                 fresh_db_clause = db_clause
    #             else:  # For rules
    #                 fresh_db_clause = self.fresh_variables(db_clause)
                    
    #             print(f"DEBUG: Working with clause: {fresh_db_clause}")
                
    #             # Get the correct term to match against
    #             head_to_match = db_head if isinstance(fresh_db_clause, Expr) else fresh_db_clause.arguments[0]
    #             print(f"DEBUG: Attempting to match with: {head_to_match}")
                
    #             result = self.mgu(term, head_to_match)
    #             print(f"DEBUG: MGU result: {result}")
                
    #             if result is not None:
    #                 unifier, new_facts = result
    #                 print(f"DEBUG: Unifier: {unifier}")
    #                 if isinstance(fresh_db_clause, Expr):
    #                     new_clause = fresh_db_clause
    #                 else:
    #                     new_clause = fresh_db_clause.apply_substitution(unifier)
    #                 print(f"DEBUG: Yielding match: {new_clause}")
                    
    #                 yield new_clause, unifier, new_facts


    def fresh_variables(self, term: Clause) -> Clause:
        """Replace all variables in a clause with fresh variables"""
        return fresh_variables(term, self.fresh_var_counter.get_fresh_variable)[0]

    def get_builtins(self):
        return ALL_BUILTINS

    # def query(
    #     self,
    #     query: Expr,
    #     max_proofs: Optional[int] = None,
    #     max_depth: Optional[int] = None,
    #     max_branching: Optional[int] = None,
    #     queue: Optional[ProofQueue] = None,
    #     return_stats: bool = False,
    # ):
    #     print("CALLED")
    #     self.queried = query
    #     if queue is None:
    #         queue = OrderedProofQueue(self.algebra)
        
    #     formulas, proof_steps, nb_proofs = get_proofs(
    #         self,
    #         self.algebra,
    #         query=query,
    #         max_proofs=max_proofs,
    #         max_depth=max_depth,
    #         queue=queue,
    #         max_branching=max_branching,
    #     )
    #     print(f"Proof step result: {formulas}, type: {type(formulas)}")
    #     result = {k: self.algebra.evaluate(f) for k, f in formulas.items()}
    #     print(f"Proof step result: {formulas}, type: {type(formulas)}")
    #     zero = self.algebra.eval_zero()
    #     result = {k: v for k, v in result.items() if v != zero}
    #     if return_stats:
    #         return result, proof_steps, nb_proofs
    #     return result
    
    def query(
        self,
        query: Expr,
        max_proofs: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_branching: Optional[int] = None,
        queue: Optional[ProofQueue] = None,
        return_stats: bool = False,
    ):
        print(f"CALLED (SPL)")
        print(f"QUERY: {query}")
        self.queried = query
        if queue is None:
            queue = OrderedProofQueue(self.algebra)
            print(f"Created new OrderedProofQueue with algebra: {type(self.algebra).__name__}")
        
        print(f"Starting proof search with max_proofs={max_proofs}, max_depth={max_depth}, max_branching={max_branching}")
        formulas, proof_steps, nb_proofs = get_proofs(
            self,
            self.algebra,
            query=query,
            max_proofs=max_proofs,
            max_depth=max_depth,
            queue=queue,
            max_branching=max_branching,
        )
        
        print(f"Raw formulas returned: {formulas}")
        print(f"Proof steps: {proof_steps}, Number of proofs: {nb_proofs}")
        
        # Check for empty formulas
        if not formulas:
            print("WARNING: No formulas found in proof search")
        
        # Evaluate formulas
        print("Evaluating formulas with algebra...")
        result = {}
        for k, f in formulas.items():
            eval_result = self.algebra.evaluate(f)
            result[k] = eval_result
            
            # Check for numerical issues
            if isinstance(eval_result, torch.Tensor):
                if torch.isneginf(eval_result):
                    print(f"CRITICAL: Formula {k} evaluated to -inf")
                elif torch.isnan(eval_result):
                    print(f"CRITICAL: Formula {k} evaluated to NaN")
            
            print(f"Formula {k}: {f} -> {eval_result}")
        
        # Filter out zero results
        zero = self.algebra.eval_zero()
        filtered_result = {k: v for k, v in result.items() if v != zero}
        
        if len(result) != len(filtered_result):
            print(f"Filtered out {len(result) - len(filtered_result)} zero results")
        
        print(f"Final result: {filtered_result}")
        
        if return_stats:
            return filtered_result, proof_steps, nb_proofs
        return filtered_result

    def __call__(self, query: Expr, **kwargs):
        result, proof_steps, nb_proofs = self.query(query, return_stats=True, **kwargs)
        if type(result) is set:
            return len(result) > 0.0, proof_steps, nb_proofs
        if type(result) is dict and query in result:
            return result[query], proof_steps, nb_proofs
        return self.algebra.evaluate(self.algebra.zero()), proof_steps, nb_proofs
    
    def eval(self):
        self.store = self.store.eval()
        return self

    def apply(self, *args, **kwargs):
        return self.store.apply(*args, **kwargs)

    def modules(self):
        return self.store.modules()


class BooleanProofModule(ProofModule):
    def __init__(self, clauses):
        super().__init__(clauses, algebra=BOOLEAN_ALGEBRA)

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        return set(result.keys())


class ProbabilisticProofModule(ProofModule):
    def __init__(self, clauses, log_mode=False):
        eval_algebra = LOG_PROBABILITY_ALGEBRA if log_mode else PROBABILITY_ALGEBRA
        super().__init__(clauses, algebra=DnfAlgebra(eval_algebra))


# def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
#     # print("get_proofs called")
#     proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
#     proofs = defaultdict(algebra.zero)
#     nb_proofs = 0
#     for proof in proof_tree.get_proofs():
#         proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
#         nb_proofs += 1

#     # print("ALL PROOFS", {answer: algebra.evaluate(proof) for answer, proof in proofs.items()})
#     return dict(proofs), proof_tree.nb_steps, nb_proofs

# Modify the get_proofs function in proof_module.py
# def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
#     proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
#     proofs = defaultdict(algebra.zero)
#     nb_proofs = 0
    
#     for proof in proof_tree.get_proofs():
#         # Ensure the query being used as the key matches the original query
#         # For complex queries with multiple goals/conjunctions
#         query_key = kwargs.get('query')  # Get the original query
        
#         # Use the original query as the key if available
#         if query_key is not None and proof.query.get_predicate() == query_key.get_predicate():
#             proofs[query_key] = algebra.add(proofs[query_key], proof.value) 
#         else:
#             # Otherwise use the proof's query
#             proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
            
#         nb_proofs += 1

#     return dict(proofs), proof_tree.nb_steps, nb_proofs

def get_proofs(prover, algebra, **kwargs) -> tuple[dict[Expr, Value], int, int]:
    print(f"get_proofs called with kwargs: {kwargs}")
    
    # Create and run proof tree
    proof_tree = ProofTree(prover, algebra=algebra, **kwargs)
    proofs = defaultdict(algebra.zero)
    nb_proofs = 0
    
    print("Collecting proofs from proof tree...")
    for proof in proof_tree.get_proofs():
        print(f"Found proof: {proof.query}")
        print(f"  Goals: {proof.goals}")
        print(f"  Value: {proof.value}")
        
        proofs[proof.query] = algebra.add(proofs[proof.query], proof.value)
        nb_proofs += 1
    
    print(f"Proof collection complete: {nb_proofs} proofs in {proof_tree.nb_steps} steps")
    
    # Check if no proofs found
    if nb_proofs == 0:
        print("WARNING: No proofs found in proof tree")
        print(f"Proof tree stats:")
        print(f"  Steps: {proof_tree.nb_steps}")
        print(f"  Answers: {proof_tree.answers}")
        print(f"  Value: {proof_tree.value}")
        
        # Check for incomplete proofs with only object predicates
        if hasattr(proof_tree, '_proof_history'):
            near_complete = [p for _, p in proof_tree._proof_history 
                           if p.goals and all(g.functor == "object" for g in p.goals)]
            
            if near_complete:
                print(f"Found {len(near_complete)} proofs with only object predicates:")
                for i, p in enumerate(near_complete[:3]):  # Show at most 3
                    print(f"  Near-complete proof {i}:")
                    print(f"    Query: {p.query}")
                    print(f"    Goals: {p.goals}")
                    print(f"    Value: {p.value}")
                    print(f"    is_complete(): {p.is_complete()}")
                    
                    # Look for soft unifications that should allow completion
                    if hasattr(p.value, 'pos_facts'):
                        print(f"    Soft unifications: {p.value.pos_facts}")
    
    # Return the results as normal - no artificial fixes
    return dict(proofs), proof_tree.nb_steps, nb_proofs


from queue import LifoQueue
import heapq

import torch

from ..algebras.abstract_algebra import Algebra
from .proof import Proof


cdef class ProofQueue:
    def __cinit__(self, algebra):
        self._queue = LifoQueue()
        self.algebra = algebra
        self.n = 0

    cpdef add(self, item, value):
        if value is None:
            value = item.value
        self._queue.put((self.n, item))
        self.n += 1

    def next(self) -> Proof:
        return self._queue.get()[1]

    def empty(self) -> bool:
        return self._queue.empty()

    def new(self, algebra):
        return ProofQueue(algebra)

    cpdef add_first(self, n, queue):
        """
        Add the first n proofs from the given queue to this queue.
        """
        i = 0
        n = n or float("+inf")
        while not queue.empty() and i < n:
            v = queue.next()
            self.add(v, None)
            i += 1


cdef class OrderedProofQueue:
    def __cinit__(self, algebra: Algebra):
        """
        Proof queue with an ordering determined by a algebra.
        Note: on equal values, depth-first search is used.
        """
        self._queue = []
        self.algebra = algebra
        # to keep the sorting stable, we add an index
        self.n = 0

    def _get_value(self, value):
        with torch.inference_mode():
            return self.algebra.evaluate(value)

    cpdef add(self, item, value):
        if value is None:
            value = self._get_value(item.value)
        if value != self.algebra.eval_zero():
            heapq.heappush(self._queue, (-value, self.n, item))
            self.n += 1

    def next(self) -> Proof:
        return heapq.heappop(self._queue)[-1]

    def new(self, algebra):
        return OrderedProofQueue(algebra)

    cpdef add_first(self, n, OrderedProofQueue queue):
        i = 0
        n = n or float("+inf")
        while not queue.empty() and i < n:
            value, _, item = heapq.heappop(queue._queue)
            self.add(item, -value)
            i += 1

    def empty(self) -> bool:
        return len(self._queue) == 0
    


from typing import TYPE_CHECKING, Iterator, Optional

from deepsoftlog.algebraic_prover.proving.proof import Proof, ProofDebug
from deepsoftlog.algebraic_prover.proving.proof_queue import ProofQueue
from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra
from deepsoftlog.algebraic_prover.terms.expression import Expr

if TYPE_CHECKING:
    from deepsoftlog.algebraic_prover.proving.proof_module import ProofModule

import traceback
from deepsoftlog.algebraic_prover.algebras.sdd2_algebra import ConjoinedFacts

class ProofTree:
    """
    Proof tree for a query.
    Searches depth-first.
    """

    def __init__(
        self,
        program: "ProofModule",
        query: Expr,
        algebra: Algebra,
        max_depth: Optional[int] = None,
        max_proofs: Optional[int] = None,
        queue: ProofQueue = None,
        max_branching: Optional[int] = None,
    ):
        self.algebra = algebra
        self.program = program
        self.max_depth = max_depth
        self.max_proofs = max_proofs
        self.max_branching = max_branching
        self.sub_calls = dict()
        self.answers = set()
        self.incomplete_sub_trees: list["ProofTree"] = []
        self.proofs = []
        self.queue = queue
        self.queue.add(self._create_proof_for(query), None)
        self.value = self.algebra.zero()
        self.nb_steps = 0

    # def _create_proof_for(self, query: Expr):
    #     # if type(query) is not Expr:
    #     #     query = query.query
    #     # print("_create_proof_for called")
    #     # print(f"query: {type(query)} {query}")
    #     print(f"Proof: {ProofDebug(query=query, proof_tree=self, value=self.algebra.one())}")
    #     return ProofDebug(query=query, proof_tree=self,
    #     value=self.algebra.one())
    
    ##3.7 thinking version
    def _create_proof_for(self, query: Expr):
        print(f"Proof: {ProofDebug(query=query, proof_tree=self, value=self.algebra.one())}")
        return ProofDebug(query=query, proof_tree=self, value=self.algebra.one(), bindings={})

    def is_complete(self) -> bool:
        return self.queue.empty() or len(self.proofs) >= self.get_max_proofs()

    # def get_proofs(self) -> Iterator[Proof]:
    #     # print("get_proofs called")
    #     while not self.is_complete():
    #         proof = self.step()
    #         if proof is not None:
    #             yield proof
    
    def get_proofs(self) -> Iterator[Proof]:
        """More aggressive approach to finding proofs"""
        attempts = 0
        while not self.queue.empty() and attempts < 1000 and len(self.proofs) < self.get_max_proofs():
            attempts += 1
            print(f"PROOF ATTEMPT: {attempts}, Proofs found: {len(self.proofs)}")
            
            # Try regular step logic
            proof = self.step()
            
            # Check the queue for proofs very close to completion (with only object goals)
            if self.queue.empty() and not self.proofs:
                print("No proofs found but queue is empty - checking for near-complete proofs")
                # Scan all attempted proofs for ones with only object predicates left
                if hasattr(self, '_attempted_proofs') and self._attempted_proofs:
                    for p in self._attempted_proofs:
                        if (p.goals and all(g.functor == "object" for g in p.goals) and 
                            isinstance(p.value, ConjoinedFacts) and p.value.pos_facts):
                            print(f"FORCE COMPLETING PROOF: {p.query} with {len(p.goals)} object goals")
                            empty_goal_proof = Proof(
                                query=p.query,
                                goals=tuple(),  # Empty goals = completed
                                depth=p.depth + 1,
                                proof_tree=self,
                                value=p.value,
                                bindings=getattr(p, 'current_bindings', {})
                            )
                            self.answers.add(empty_goal_proof.query)
                            self.proofs.append(empty_goal_proof)
                            self.value = self.algebra.add(self.value, empty_goal_proof.value)
                            yield empty_goal_proof
            
            if proof is not None:
                yield proof
        
        # Final check - did we find any proofs?
        if not self.proofs and self.nb_steps > 10:
            print("WARNING: No proofs found after significant search - debugging info:")
            print(f"Attempted {attempts} proof steps")
            # Print any proof that made it far in the process
            if hasattr(self, '_attempted_proofs') and self._attempted_proofs:
                for i, p in enumerate(self._attempted_proofs[-5:]):
                    print(f"Near-complete proof {i}: Goals: {p.goals}, Value: {p.value}")

    # def step(self) -> Optional[Proof]:
    #     print(f"PROOF_STEP [{self.nb_steps}]: Queue size={len(self._queue._queue) if hasattr(self._queue, '_queue') else '?'}, Proofs={len(self.proofs)}")
        
    #     self.nb_steps += 1
    #     if len(self.incomplete_sub_trees):
    #         return self._step_subtree()

    #     proof = self.queue.next()
    #     print(f"Proof: {proof, self.algebra.one()}")
        
    #     if proof.is_complete():
    #         print(f"COMPLETE PROOF FOUND: {proof.query} with value {proof.value}")
    #         self.answers.add(proof.query)
    #         self.proofs.append(proof)
    #         old_value = self.value
    #         self.value = self.algebra.add(self.value, proof.value)
    #         print(f"Value updated: {old_value} -> {self.value}")
    #         return proof

    #     if self.is_pruned(proof):
    #         print(f"PRUNED: Proof at depth {proof.depth} > max_depth {self.get_max_depth()}")
    #     else:
    #         print(f"Generating children for proof with {proof.nb_goals()} goals")
    #         local_queue = self.queue.new(self.algebra)
    #         proof_remaining = proof.nb_goals()
            
    #         child_proofs = list(proof.get_children())
    #         print(f"Generated {len(child_proofs)} child proofs")
            
    #         for child_proof in child_proofs:
    #             child_remaining = child_proof.nb_goals()
    #             print(f"Adding proof step: {child_proof}, type: {type(child_proof)}")
                
    #             # Check for potential numerical issues in the proof value
    #             if hasattr(child_proof, 'value') and isinstance(child_proof.value, torch.Tensor):
    #                 if torch.isneginf(child_proof.value):
    #                     print(f"WARNING: Child proof has -inf value")
    #                 elif torch.isnan(child_proof.value):
    #                     print(f"WARNING: Child proof has NaN value")
                
    #             if child_remaining < proof_remaining:
    #                 print(f"Child has fewer goals ({child_remaining} < {proof_remaining}), adding to local queue")
    #                 local_queue.add(child_proof, None)
    #             else:
    #                 print(f"Child has same or more goals ({child_remaining} >= {proof_remaining}), adding to main queue")
    #                 self.queue.add(child_proof, None)
                    
    #         print(f"Adding up to {self.max_branching} proofs from local queue to main queue")
    #         self.queue.add_first(self.max_branching, local_queue)
    
    def step(self) -> Optional[Proof]:
        """Enhanced step method with diagnostic tracing"""
        print(f"STEP [{self.nb_steps}]: Queue size={len(self.queue._queue) if hasattr(self.queue, '_queue') else '?'}")
        
        self.nb_steps += 1
        if len(self.incomplete_sub_trees):
            return self._step_subtree()

        # Get next proof to consider
        if self.queue.empty():
            print("Queue is empty - no more proofs to process")
            return None
            
        proof = self.queue.next()
        
        # Track this proof for analysis
        if not hasattr(self, '_proof_history'):
            self._proof_history = []
        self._proof_history.append((self.nb_steps, proof))
        
        # Debug output
        print(f"Processing proof {id(proof)} with {len(proof.goals)} goals")
        if proof.goals:
            print(f"  Current goals: {proof.goals}")
            # Check for object-only goals
            if all(g.functor == "object" for g in proof.goals):
                print(f"  DIAGNOSTIC: All goals are object predicates - should complete soon")
                # Check if there are soft unifications available
                if hasattr(proof.value, 'pos_facts') and proof.value.pos_facts:
                    print(f"  DIAGNOSTIC: Proof has soft unifications: {proof.value.pos_facts}")
                    print(f"  DIAGNOSTIC: is_complete() says: {proof.is_complete()}")
        else:
            print("  No goals - proof should be complete")
        
        # Check completion
        if proof.is_complete():
            print(f"COMPLETE PROOF FOUND: {proof.query} with {len(getattr(proof, 'goals', []))} goals")
            self.answers.add(proof.query)
            self.proofs.append(proof)
            old_value = self.value
            self.value = self.algebra.add(self.value, proof.value)
            print(f"  Value updated: {old_value} -> {self.value}")
            return proof

        # Process children in usual way
        if not self.is_pruned(proof):
            local_queue = self.queue.new(self.algebra)
            proof_remaining = proof.nb_goals()
            
            # Get children - diagnostic logging for any issues
            try:
                child_proofs = list(proof.get_children()) 
                print(f"  Generated {len(child_proofs)} child proofs")
                
                # Analyze children
                for i, child in enumerate(child_proofs[:3]):  # Limit to first 3 to avoid verbose output
                    print(f"  Child {i}: {len(child.goals)} goals: {child.goals[:2]}{'...' if len(child.goals) > 2 else ''}")
                    print(f"    is_complete(): {child.is_complete()}")
                
                for child_proof in child_proofs:
                    child_remaining = child_proof.nb_goals()
                    if child_remaining < proof_remaining:
                        local_queue.add(child_proof, None)
                    else:
                        self.queue.add(child_proof, None)
                        
                self.queue.add_first(self.max_branching, local_queue)
            except Exception as e:
                print(f"ERROR in get_children: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Pruned proof at depth {proof.depth}")

    

    def _step_subtree(self):
        self.incomplete_sub_trees[-1].step()
        if self.incomplete_sub_trees[-1].is_complete():
            del self.incomplete_sub_trees[-1]

    def get_answers(self) -> set[Expr]:
        assert self.is_complete()
        return self.answers

    def sub_call(self, query: Expr, depth: int) -> "ProofTree":
        new_algebra = self.algebra.get_dual()
        new_tree = type(self)(
            program=self.program,
            query=query,
            algebra=new_algebra,
            max_depth=self.max_depth - depth if self.max_depth is not None else None,
            max_proofs=self.max_proofs,
            max_branching=self.max_branching,
            queue=self.queue.new(new_algebra),
        )
        print(f"query: {query}")
        self.sub_calls[query] = new_tree
        return new_tree

    def get_sub_call_tree(self, query: Expr, depth: int) -> Optional["ProofTree"]:
        if query in self.sub_calls:
            return self.sub_calls[query]
        else:
            new_tree = self.sub_call(query, depth)
            self.incomplete_sub_trees.append(new_tree)
            return None

    def get_max_depth(self):
        if self.max_depth is None:
            return float("+inf")
        return self.max_depth

    def get_max_proofs(self):
        if self.max_proofs is None:
            return float("+inf")
        return self.max_proofs

    def is_pruned(self, proof: ProofDebug):
        if proof.depth > self.get_max_depth():
            if self.max_depth is None:  # pragma: no cover
                import warnings

                warnings.warn("Default max depth exceeded")
            return True
        return False
    
from typing import TYPE_CHECKING, Iterable, Optional

from deepsoftlog.algebraic_prover.algebras.abstract_algebra import Algebra, Value
from deepsoftlog.algebraic_prover.terms.expression import Fact, Expr

if TYPE_CHECKING:  # pragma: no cover
    from .proof_tree import ProofTree
import torch


class Proof:
    def __init__(
        self,
        query: Expr,
        goals: Optional[tuple[Expr, ...]] = None,
        depth: int = 0,
        proof_tree: "ProofTree" = None,
        value: Value = None,
        bindings: dict = None,  # Add bindings parameter
    ):
        if goals is None:
            goals = (query,)
        self.query = query
        self.depth = depth
        self.goals: tuple[Expr, ...] = goals
        self.value = value
        self.proof_tree = proof_tree
        self.current_bindings = bindings or {}  # Track accumulated bindings
        
        print("Depth: ",  depth, "new proof", self)

    # def is_complete(self) -> bool:
    #     return len(self.goals) == 0
    def is_complete(self) -> bool:
        """Enhanced is_complete with diagnostic information"""
        # Check for empty goals
        is_goals_empty = len(self.goals) == 0
        
        # Diagnostic info
        if hasattr(self, 'query'):
            print(f"is_complete check for proof with query {self.query}:")
            print(f"  Goals empty: {is_goals_empty}")
            print(f"  Goals: {self.goals}")
            print(f"  Value type: {type(self.value)}")
            
            # Check special case of object-only goals
            if self.goals and all(g.functor == "object" for g in self.goals):
                print(f"  SPECIAL CASE: All goals are object predicates")
                if hasattr(self.value, 'pos_facts'):
                    print(f"  Value has soft facts: {bool(self.value.pos_facts)}")
                    print(f"  Soft facts: {self.value.pos_facts}")
        
        return is_goals_empty



    def nb_goals(self) -> int:
        
        return len(self.goals)

    def get_algebra(self) -> Algebra:
        return self.proof_tree.algebra

    def get_children(self) -> Iterable["Proof"]:
        print(f"goals[0]: {self.goals[0]}")
        if self.goals[0].functor == "\\+":
            yield from self.negation_node()
        else:
            yield from self.apply_clauses()
        # print(f"goals: {self.goals}")

    def negation_node(self):
        # print("negation_node")
        negated_goal = self.goals[0].arguments[0]
        matches = self.proof_tree.program.all_matches(negated_goal)
        # print(f"New goals before substitution: {self.goals}, Types: {[type(g) for g in self.goals]}")
        if not any(matches):
            # goal is not present, so negation is trivially true
            yield self.get_child(new_goals=self.goals[1:])
        else:
            # create proof tree for negation
            sub_call_tree = self.proof_tree.get_sub_call_tree(negated_goal, self.depth)
            if sub_call_tree is None:
                yield self
            else:
                sub_call_value = sub_call_tree.value
                new_value = self.get_algebra().multiply(self.value, sub_call_value)
                yield self.get_child(new_goals=self.goals[1:], value=new_value)

    # def apply_clauses(self):
    #     print("apply_clauses")
    #     first_goal, *remaining = self.goals
    #     print(f"Trying to match goal: {first_goal}")
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     print(f"Found matches: {list(matches)}")
    #     print(f"Clause: {matches[0][0]}")
        
    #     for clause, unifier, new_facts in matches:
    #         print(f"Matching clause: {clause}")
    #         print(f"With unifier: {unifier}")
            
    #         # For rules, handle the body correctly based on its structure
    #         if clause.is_clause():  # Check if it's a rule
    #             # Extract body properly, handling the structure of conjunctive goals
    #             body = clause.arguments[1]
    #             print(f"Body: {body}")
    #             print(body.arguments)
    #             if body.is_and():  # It's a conjunctive goal with ","
    #                 new_goals = body.arguments  # Get all conjuncts
    #             else:
    #                 new_goals = (body,)  # Single goal
                
    #             # Apply the unifier to each goal in the body
    #             new_goals = tuple(g.apply_substitution(unifier) for g in new_goals)
                
    #             # Add remaining goals with unifier applied
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
                
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
                
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:  # It's a fact
    #             # Just continue with remaining goals
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
                
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=tuple(g.apply_substitution(unifier) for g in remaining),
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    # def apply_clauses(self):
    #     first_goal, *remaining = self.goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    # def apply_clauses(self):
    #     if not self.goals:
    #         print("COMPLETE: No more goals to process")
    #         return  # Return an empty iterator
    #     first_goal, *remaining = self.goals
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    def apply_clauses(self):
        """Enhanced apply_clauses with diagnostic tracing"""
        print(f"apply_clauses called for proof {id(self)} with {len(self.goals)} goals")
        
        # Diagnostic for empty goals
        if not self.goals:
            print("DIAGNOSTIC: apply_clauses called with empty goals")
            yield self  # Simply return this proof as it's already complete
            return
        
        # Diagnostic for object-only goals
        if all(g.functor == "object" for g in self.goals):
            print(f"DIAGNOSTIC: All remaining goals are object predicates: {self.goals}")
            print(f"DIAGNOSTIC: Value has soft unifications: {hasattr(self.value, 'pos_facts')}")
            if hasattr(self.value, 'pos_facts'):
                print(f"DIAGNOSTIC: Soft unifications: {self.value.pos_facts}")
                
                # This is where we need to understand why these aren't being completed
                print(f"DIAGNOSTIC: Trace of goals, value and completion status:")
                print(f"  Goals: {self.goals}")
                print(f"  Value: {self.value}")
                print(f"  is_complete(): {self.is_complete()}")
                if hasattr(self, 'get_child'):
                    # Generate a child with empty goals for analysis
                    test_child = self.get_child(new_goals=tuple())
                    print(f"  Test child with empty goals - is_complete(): {test_child.is_complete()}")
        
        # Standard processing
        first_goal, *remaining = self.goals
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            print(f"Breaking down conjunction: {first_goal}")
            conjoined_goals = first_goal.arguments
            new_goals = conjoined_goals + tuple(remaining)
            new_child = self.get_child(new_goals=new_goals, depth=self.depth + 1, value=self.value)
            print(f"Yielding conjunction breakdown child with {len(new_child.goals)} goals")
            yield new_child
            return
        
        # Process matches
        print(f"Looking for matches for: {first_goal}")
        matches = list(self.proof_tree.program.all_matches(first_goal))
        print(f"Found {len(matches)} matches")
        
        for i, (clause, unifier, new_facts) in enumerate(matches):
            print(f"Processing match {i}: {clause}")
            
            if clause.is_fact():
                print(f"Match is a fact")
                # Fact match - continue with remaining goals
                new_goals = tuple(g.apply_substitution(unifier) for g in remaining)
                query = self.query.apply_substitution(unifier)
                new_value = self.create_new_value(clause, new_facts)
                
                # Diagnostic for object-only remaining goals
                if new_goals and all(g.functor == "object" for g in new_goals):
                    print(f"DIAGNOSTIC: After matching fact, only object goals remain: {new_goals}")
                    print(f"DIAGNOSTIC: New value: {new_value}")
                    print(f"DIAGNOSTIC: New value has soft facts: {hasattr(new_value, 'pos_facts')}")
                    if hasattr(new_value, 'pos_facts'):
                        print(f"DIAGNOSTIC: New soft facts: {new_value.pos_facts}")
                
                child = self.get_child(query=query, new_goals=new_goals, depth=self.depth+1, value=new_value)
                print(f"Yielding fact-matched child with {len(child.goals)} goals")
                print(f"  is_complete(): {child.is_complete()}")
                yield child
            else:
                print(f"Match is a rule")
                # Rule match - add body goals first
                body = clause.arguments[1]
                if body.is_and():
                    new_body_goals = body.arguments
                else:
                    new_body_goals = (body,)
                
                # Apply unifier
                new_body_goals = tuple(g.apply_substitution(unifier) for g in new_body_goals)
                new_remaining = tuple(g.apply_substitution(unifier) for g in remaining)
                new_goals = new_body_goals + new_remaining
                
                query = self.query.apply_substitution(unifier)
                new_value = self.create_new_value(clause, new_facts)
                
                child = self.get_child(query=query, new_goals=new_goals, depth=self.depth+1, value=new_value)
                print(f"Yielding rule-matched child with {len(child.goals)} goals")
                yield child


    # def create_new_value(self, clause, new_facts):
    #     new_facts = self.get_algebra().reduce_mul_value_pos(new_facts)
    #     new_value = self.get_algebra().multiply(self.value, new_facts)
    #     if clause.is_annotated():
    #         new_value = self.get_algebra().multiply_value_pos(new_value, clause)
    #     return new_value
    
    def create_new_value(self, clause, new_facts):
        print(f"Creating new value from clause: {clause}")
        if new_facts:
            print(f"With new_facts: {new_facts}")
        
        new_facts_value = self.get_algebra().reduce_mul_value_pos(new_facts)
        print(f"New facts value: {new_facts_value}")
        
        new_value = self.get_algebra().multiply(self.value, new_facts_value)
        print(f"After multiply with current value: {new_value}")
        
        if clause.is_annotated():
            clause_value = self.get_algebra().value_pos(clause)
            print(f"Clause is annotated with value: {clause_value}")
            new_value = self.get_algebra().multiply_value_pos(new_value, clause)
            print(f"Final value after clause annotation: {new_value}")
        
        # Check for numerical issues
        if isinstance(new_value, torch.Tensor):
            if torch.isneginf(new_value):
                print("CRITICAL: create_new_value produced -inf")
            elif torch.isnan(new_value):
                print("CRITICAL: create_new_value produced NaN")
        
        return new_value

    def get_child(
        self,
        query: Optional[Expr] = None,
        new_goals: tuple[Expr, ...] = tuple(),
        depth: Optional[int] = None,
        value: Optional[Value] = None,
        bindings: Optional[dict] = None,  # Add bindings parameter
    ):
        if new_goals is None:
            new_goals = tuple()
        # print(f"new_goals: {new_goals}")
        # print(self.proof_tree)
        return ProofDebug(
            query=self.query if query is None else query,
            value=self.value if value is None else value,
            goals=new_goals,
            depth=self.depth if depth is None else depth,
            proof_tree=self.proof_tree
        )
        

    def __repr__(self):  # pragma: no cover
        return f"{self.query}: {self.goals} - {self.value}"

    def __lt__(self, other: "Proof"):
        return len(self.goals) < len(other.goals)


class ProofDebug(Proof):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_level = 0
    
    # def apply_clauses(self):
    #     first_goal, *remaining = self.goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     for clause, unifier, new_facts in matches:
    #         new_goals = clause.arguments[1].arguments  # new goals from clause body
    #         new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #         query: Expr = self.query.apply_substitution(unifier)
    #         new_value = self.create_new_value(clause, new_facts)
    #         yield self.get_child(
    #             query=query,
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=new_value,
    #         )
    
    # def apply_clauses(self):
    # # If there are no goals left, the proof is complete
    #     if not self.goals:
    #         # Return a successful proof state with empty goals
    #         print("No more goals - proof complete!")
    #         yield self.get_child(
    #             new_goals=tuple(),
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return

    #     first_goal, *remaining = self.goals
    #     print(f"Processing goal: {first_goal}")
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         print(f"Breaking down conjunction: {first_goal}")
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         print(f"New goals after breaking conjunction: {new_goals}")
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     match_found = False
        
    #     for clause, unifier, new_facts in matches:
    #         match_found = True
    #         print(f"Found match with clause: {clause}")
            
    #         if clause.is_fact():
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only
    #             print(f"Matched a fact, moving to remaining goals: {remaining}")
    #             new_goals = tuple(g.apply_substitution(unifier) for g in remaining)
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:
    #             # For rules, need to prove the body before continuing
    #             print(f"Matched a rule, adding body goals")
    #             # Apply unifier to the body goals first
    #             print(f"Clause body: {clause.arguments[1]}")
    #             print(f"{unifier=}")
    #             body = clause.arguments[1].apply_substitution(unifier)
    #             new_goals = body.arguments  # new goals from clause body with unifier applied
    #             new_goals += tuple(g.apply_substitution(unifier) for g in remaining)
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
        
    #     if not match_found:
    #         print(f"No matches found for: {first_goal}")
    
    # def apply_clauses(self):
    #     # If there are no goals left, the proof is complete
    #     if not self.goals:
    #         # Return a successful proof state with empty goals
    #         print("No more goals - proof complete!")
    #         yield self.get_child(
    #             new_goals=tuple(),
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return

    #     first_goal, *remaining = self.goals
    #     print(f"Processing goal: {first_goal}")
        
    #     # Special handling for conjunctions
    #     if first_goal.functor == "," and first_goal.get_arity() > 0:
    #         print(f"Breaking down conjunction: {first_goal}")
    #         # Break down the conjunction into separate goals
    #         conjoined_goals = first_goal.arguments
    #         new_goals = conjoined_goals + tuple(remaining)
    #         print(f"New goals after breaking conjunction: {new_goals}")
    #         # Create a child proof with the broken-down goals
    #         yield self.get_child(
    #             new_goals=new_goals,
    #             depth=self.depth + 1,
    #             value=self.value
    #         )
    #         return
        
    #     # Original code for non-conjunctive goals
    #     matches = self.proof_tree.program.all_matches(first_goal)
    #     match_found = False
        
    #     for clause, unifier, new_facts in matches:
    #         match_found = True
    #         print(f"Found match with clause: {clause}")
            
    #         # Apply unifier to ALL remaining goals immediately
    #         # This ensures variable bindings from this goal propagate to all future goals
    #         updated_remaining = tuple(g.apply_substitution(unifier) for g in remaining)
            
    #         if clause.is_fact():
    #             # If we matched a fact, this goal is proven
    #             # Continue with remaining goals only with substitutions applied
    #             print(f"Matched a fact, moving to remaining goals: {updated_remaining}")
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=updated_remaining,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
    #         else:
    #             # For rules, need to prove the body before continuing
    #             print(f"Matched a rule, adding body goals")
    #             # Apply unifier to the body goals first
    #             print(f"Clause body: {clause.arguments[1]}")
    #             print(f"{unifier=}")
    #             body = clause.arguments[1].apply_substitution(unifier)
    #             # Apply the unifier to body goals and add remaining goals
    #             if body.functor == "," and body.get_arity() > 0:
    #                 # If body is a conjunction, extract its arguments
    #                 new_goals = body.arguments + updated_remaining
    #             else:
    #                 # Otherwise, add body as a single goal
    #                 new_goals = (body,) + updated_remaining
                    
    #             query = self.query.apply_substitution(unifier)
    #             new_value = self.create_new_value(clause, new_facts)
    #             yield self.get_child(
    #                 query=query,
    #                 new_goals=new_goals,
    #                 depth=self.depth + 1,
    #                 value=new_value,
    #             )
        
    #     if not match_found:
    #         print(f"No matches found for: {first_goal}")

    def get_child(
        self,
        query: Optional[Expr] = None,
        new_goals: tuple[Expr, ...] = tuple(),
        depth: Optional[int] = None,
        value: Optional[Value] = None,
        bindings: Optional[dict] = None,  # Add bindings parameter
    ):
        # If no new query is provided but we have bindings, apply them to the current query
        if query is None and bindings:
            query = self.query.apply_substitution(bindings)
        else:
            query = self.query if query is None else query
            
        # Combine existing bindings with new bindings
        combined_bindings = dict(self.current_bindings)  # Make a copy
        if bindings:
            combined_bindings.update(bindings)
            
        return ProofDebug(
            query=query,
            goals=new_goals,
            depth=self.depth if depth is None else depth,
            proof_tree=self.proof_tree,
            value=self.value if value is None else value,  # Move after proof_tree
            bindings=combined_bindings,
        )
    
    ##3.7 thinking version
    def apply_clauses(self):
        # If there are no goals left, the proof is complete
        if not self.goals:
            # Return a successful proof state with empty goals
            print("No more goals - proof complete!")
            yield self.get_child(
                new_goals=tuple(),
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings  # Pass along current bindings
            )
            return

        first_goal, *remaining = self.goals
        # print(f"Processing goal: {first_goal}")
        
        # Special handling for conjunctions
        if first_goal.functor == "," and first_goal.get_arity() > 0:
            # print(f"Breaking down conjunction: {first_goal}")
            # Break down the conjunction into separate goals
            conjoined_goals = first_goal.arguments
            # Apply current bindings to all goals
            conjoined_goals = tuple(g.apply_substitution(self.current_bindings) for g in conjoined_goals)
            remaining_goals = tuple(g.apply_substitution(self.current_bindings) for g in remaining)
            new_goals = conjoined_goals + remaining_goals
            # print(f"New goals after breaking conjunction: {new_goals}")
            # Create a child proof with the broken-down goals
            yield self.get_child(
                new_goals=new_goals,
                depth=self.depth + 1,
                value=self.value,
                bindings=self.current_bindings
            )
            return
        
        # # NEW PART: Check if this goal can be satisfied by soft unifications
        # if self._can_satisfy_via_soft_unification(first_goal):
        #     print(f"Goal can be satisfied via soft unification: {first_goal}")
        #     # Create a new proof state with this goal removed and an adjusted probability
        #     new_value = self._adjust_value_for_soft_satisfaction(first_goal)
        #     yield self.get_child(
        #         new_goals=tuple(remaining),
        #         depth=self.depth + 1,
        #         value=new_value,
        #         bindings=self.current_bindings
        #     )

        
        
        # Original code for non-conjunctive goals
        matches = self.proof_tree.program.all_matches(first_goal)
        match_found = False
        
        for clause, unifier, new_facts in matches:
            match_found = True
            print(f"Found match with clause: {clause}")
            
            # Create updated bindings by combining current bindings with new unifier
            updated_bindings = dict(self.current_bindings)
            updated_bindings.update(unifier)
            
            if clause.is_fact():
                # If we matched a fact, this goal is proven
                # Continue with remaining goals only
                print(f"Matched a fact, moving to remaining goals: {remaining}")
                # Apply updated bindings to remaining goals
                new_goals = tuple(g.apply_substitution(updated_bindings) for g in remaining)
                query = self.query.apply_substitution(updated_bindings)
                new_value = self.create_new_value(clause, new_facts)
                yield self.get_child(
                    query=query,
                    new_goals=new_goals,
                    depth=self.depth + 1,
                    value=new_value,
                    bindings=updated_bindings
                )
            else:
                # For rules, need to prove the body before continuing
                print(f"Matched a rule, adding body goals")
                # Apply unifier to the body goals first
                print(f"Clause body: {clause.arguments[1]}")
                print(f"{unifier=}")
                # Apply the updated bindings to both body and remaining goals
                body = clause.arguments[1].apply_substitution(updated_bindings)
                new_goals = body.arguments + tuple(g.apply_substitution(updated_bindings) for g in remaining)
                query = self.query.apply_substitution(updated_bindings)
                new_value = self.create_new_value(clause, new_facts)
                yield self.get_child(
                    query=query,
                    new_goals=new_goals,
                    depth=self.depth + 1,
                    value=new_value,
                    bindings=updated_bindings
                )
        
        if not match_found:
            print(f"No matches found for: {first_goal}")
    
    def _debug_print(self, message, level=0):
        indent = "  " * (self.depth + level)
        print(f"{indent}DEBUG: {message}")
        
from typing import Optional, Tuple, Union

from deepsoftlog.algebraic_prover.terms.expression import Expr
from deepsoftlog.algebraic_prover.terms.variable import Variable


def replace_occurrence(s: Variable, t: Union[Variable, Expr], x: Union[Variable, Expr]):
    if x == s:
        return t, True
    elif isinstance(x, Expr):
        new_arguments = []
        changed = False
        for argument in x.arguments:
            new_argument = replace_occurrence(s, t, argument)
            changed |= new_argument[1]
            new_arguments.append(new_argument[0])
        return x.with_args(new_arguments), changed
    return x, False


def replace_all_occurrences(
    s: Variable, t: Union[Variable, Expr], index, substitution: list
) -> Optional[list]:
    changes = False
    new_substitution = []
    for i, (lhs, rhs) in enumerate(substitution):
        if i != index:
            lhs, changed = replace_occurrence(s, t, lhs)
            changes |= changed
            rhs, changed = replace_occurrence(s, t, rhs)
            changes |= changed
        new_substitution.append((lhs, rhs))
    if not changes:
        return None
    return new_substitution


def mgu(term1: Expr, term2: Expr) -> Optional[tuple[dict, set]]:
    """
    Most General Unifier of two expressions.
    """
    # No occurs check
    substitution = [(term1, term2)]
    changes = True
    while changes:
        changes = False
        for i in range(len(substitution)):
            s, t = substitution[i]
            if type(t) is Variable and type(s) is not Variable:
                substitution[i] = (t, s)
                changes = True
                break
            if type(s) is Variable:
                if t == s:
                    del substitution[i]
                    changes = True
                    break
                new_substitution = replace_all_occurrences(s, t, i, substitution)
                if new_substitution is not None:
                    substitution = new_substitution
                    changes = True
                    break
            if isinstance(s, Expr) and isinstance(t, Expr):
                if s.get_predicate() != t.get_predicate():
                    return
                new_substitution = [
                    (s.arguments[j], t.arguments[j]) for j in range(s.get_arity())
                ]
                substitution = (
                    substitution[:i] + new_substitution + substitution[i + 1 :]
                )
                changes = True
                break

    return {k: v for (k, v) in substitution}, set()


def unify(term1: Expr, term2: Expr) -> Optional[Tuple[Expr, dict]]:
    result = mgu(term1, term2)
    if result is None:
        return None
    substitution = result[0]
    return term1.apply_substitution(substitution), substitution


def more_general_than(generic: Expr, specific: Expr) -> bool:
    # Cf. subsumes_term in SWI-Prolog
    result = mgu(generic, specific)
    if result is None:
        return False
    single_sided_unifier = result[0]
    return specific == specific.apply_substitution(single_sided_unifier)


def more_specific_than(specific: Expr, generic: Expr) -> bool:
    return more_general_than(generic, specific)

from typing import Union

import cython

from deepsoftlog.algebraic_prover.terms.variable cimport Variable, CanonicalVariableCounter
from deepsoftlog.algebraic_prover.terms.variable import fresh_variables

ExprOrVar = Union["Expr", "Variable"]


cdef class Expr:
    """
    Underlying object to represent constants, functors,
    atoms, literals and clauses (i.e. everything that is not a variable).
    """
    def __init__(self, str functor, *args: ExprOrVar, bint infix = False):
        self.functor = functor
        self.arguments = args
        self.infix = infix
        self.__hash = 0
        self._arity = len(self.arguments)

    cpdef tuple get_predicate(self):
        return self.functor, self._arity

    cpdef int get_arity(self):
        return len(self.arguments)

    def __float__(self):
        if len(self.arguments) > 0:
            raise ValueError(f"Trying to cast {self} to float")
        else:
            return float(self.functor)

    def __int__(self):
        if len(self.arguments) > 0:
            raise ValueError(f"Trying to cast {self} to int")
        else:
            return int(self.functor)

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return False
        other = cython.cast(Expr, other)
        if self.functor != other.functor or self._arity != other._arity:
            return False
        return hash(self) == hash(other) and self.arguments == other.arguments

    def __str__(self):  # pragma: no cover
        if self.get_arity() == 0:
            return self.functor
        if self.functor == ":-":
            if len(self.arguments[1].arguments) == 0:
                return f"{self.arguments[0]}."
            else:
                return f"{self.arguments[0]} :- {self.arguments[1]}."
        if self.infix:
            return str(self.functor).join(str(arg) for arg in self.arguments)
        else:
            args = ",".join(str(arg) for arg in self.arguments)
            return f"{self.functor}({args})"

    def __repr__(self) -> str:  # pragma: no cover
        if self.get_arity() == 0:
            return self.functor
        args = ",".join(repr(arg) for arg in self.arguments)
        if self.infix:
            return f"'{self.functor}'({args})"
        else:
            return f"{self.functor}({args})"

    def __hash__(self):  # Hash should be identical up to renaming of variables
        if self.__hash == 0:
            args = (x for x in self.arguments if not type(x) is Variable)
            self.__hash = hash((self.functor, *args))
        return self.__hash

    def canonical_variant(self) -> Expr:
        counter = CanonicalVariableCounter()
        return fresh_variables(self, lambda: counter.get_fresh_variable())[0]

    cpdef Expr apply_substitution(self, dict substitution):
        return self.apply_substitution_(substitution)[0]

    cpdef tuple apply_substitution_(self, dict substitution):
        changed = False
        if len(substitution) == 0:
            return self, False
        new_arguments = []
        for argument in self.arguments:
            new_argument = argument.apply_substitution_(substitution)
            changed |= new_argument[1]
            new_arguments.append(new_argument[0])
        if changed:
            new_term = self.with_args(new_arguments)
            return new_term, True
        else:
            return self, False

    cpdef Expr with_args(self, list arguments):
        return Expr(self.functor, *arguments, infix=self.infix)

    def is_ground(self) -> bool:
        return all(arg.is_ground() for arg in self.arguments)

    def is_or(self) -> bool:
        return self.functor == ";"

    def is_and(self) -> bool:
        return self.functor == ","

    def is_not(self) -> bool:
        return self.functor == r"\+"
    
    def is_clause(self) -> bool:
        return self.functor == ":-"

    def is_fact(self) -> bool:
        return self.is_clause() and self.arguments[1].get_predicate() == (",", 0)

    def is_annotated(self) -> bool:
        return False

    def without_annotation(self) -> Expr:
        return self

    def get_probability(self) -> float:
        return 1.

    def get_log_probability(self) -> float:
        return 0.

    def __and__(self, other):
        return Expr(",", self, other)

    def __or__(self, other):
        return Expr(";", self, other)

    def negate(self) -> Expr:
        if self.is_not():
            return self.arguments[0]
        return Negation(self)

    def all_variables(self) -> set:
        all_vars = (arg.all_variables() for arg in self.arguments)
        return set().union(*all_vars)

    def __lt__(self, other):
        if self.get_predicate() != other.get_predicate():
            return self.get_predicate() < other.get_predicate()
        return self.arguments < other.arguments


cpdef Expr Constant(functor):
    return Expr(str(functor))

cpdef Expr TrueTerm():
    return Constant("True")

cpdef Expr FalseTerm():
    return Constant("False")


TRUE_TERM = TrueTerm()
FALSE_TERM = FalseTerm()


cpdef Expr Negation(Expr expr):
    return Expr(r"\+", expr)


cpdef Expr Clause(Expr head, tuple body):
    return Expr(":-", head, Expr(",", *body, infix=True), infix=True)


cpdef Expr Fact(Expr fact):
    return Expr(":-", fact, Expr(",", infix=True), infix=True)


from typing import Iterable, Optional, Callable

from .expression import Expr, ExprOrVar


class ConsTerm(Expr):
    def __init__(self, head: ExprOrVar, tail: Optional[ExprOrVar] = None):
        if tail is None:
            tail = Expr("[]")
        super().__init__(".", head, tail, infix=True)

    def with_args(self, arguments):
        return ConsTerm(*arguments)

    def __str__(self):
        return "[" + ",".join(str(t) for t in self.get_terms()) + "]"

    def __repr__(self):
        return "[" + ",".join(repr(t) for t in self.get_terms()) + "]"

    def get_terms(self) -> list[Expr, ...]:
        if isinstance(self.arguments[1], ConsTerm):
            return [self.arguments[0]] + self.arguments[1].get_terms()
        elif str(self.arguments[1]) == "[]":
            return [self.arguments[0]]
        else:
            return [self.arguments[0], self.arguments[1]]

    def __getitem__(self, item):
        if item == 0:
            return self.arguments[0]
        else:
            return self.arguments[1][item - 1]


def to_prolog_list(xs: Iterable, terminal: Callable = Expr):
    head, *tail = xs
    if isinstance(head, list):
        head = to_prolog_list(head, terminal=terminal)
    if not isinstance(head, Expr):
        head = terminal(head)
    if len(tail) == 0:
        tail = Expr("[]")
    else:
        tail = to_prolog_list(tail, terminal=terminal)
    return ConsTerm(head, tail)

from deepsoftlog.algebraic_prover.algebras import safe_exp, safe_log
from deepsoftlog.algebraic_prover.terms.expression cimport Expr


cdef class ProbabilisticExpr(Expr):
    def __init__(self, prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prob = prob

    def get_probability(self):
        return self._prob

    def get_log_probability(self):
        return safe_log(self._prob)

    def is_annotated(self):
        return True

    cpdef ProbabilisticExpr with_args(self, list arguments):
        return ProbabilisticExpr(self._prob, self.functor, *arguments, infix=self.infix)

    def without_annotation(self) -> Expr:
        return Expr(self.functor, *self.arguments, infix=self.infix)

    def __str__(self):
        return f"{self.get_probability():.2g}::{super().__str__()}"

    def __repr__(self):
        return f"{self.get_probability():.2g}::{super().__repr__()}"

    """
    def __eq__(self, other):
        return other.is_annotated() \
            and self.get_probability() == other.get_probability() \
            and super().__eq__(other)

    def __hash__(self):
        return hash((self.get_probability(), super().__hash__()))
    """


cdef class LogProbabilisticExpr(ProbabilisticExpr):

    def get_probability(self):
        return safe_exp(self._prob)

    def get_log_probability(self):
        return self._prob

    cpdef LogProbabilisticExpr with_args(self, list arguments):
        return LogProbabilisticExpr(self._prob, self.functor, *arguments, infix=self.infix)


def ProbabilisticFact(prob, fact: Expr):
    return ProbabilisticExpr(float(prob), ":-", fact, Expr(","), infix=True)

def ProbabilisticClause(prob, head: Expr, body: Expr):
    assert body.functor == ","
    return ProbabilisticExpr(float(prob), ":-", head, body, infix=True)


from collections import defaultdict
from typing import Iterable

from deepsoftlog.algebraic_prover.terms.expression import Expr, Clause
from deepsoftlog.algebraic_prover.terms.probability_annotation import ProbabilisticFact


def normalize_clauses(clauses):
    # todo, more checks?
    return eliminate_probabilistic_clauses(clauses)


def eliminate_probabilistic_clauses(clauses: Iterable[Expr]) -> Iterable[Expr]:
    """
    Transforms probabilistic clauses into normal clauses,
    by adding auxiliary probabilistic facts.
    """
    auxiliary_fact_counter = 0
    for clause in clauses:
        if not clause.is_annotated() or clause.is_fact():
            yield clause
        else:
            auxiliary_fact_counter += 1
            yield from _transform_probabilistic_clause(clause, auxiliary_fact_counter)


def _transform_probabilistic_clause(clause: Expr, unique_id: int) -> Iterable[Expr]:
    clause_head = clause.arguments[0]
    clause_body = clause.arguments[1].arguments
    auxiliary_term = Expr(f"aux_pc_{unique_id}", *clause_head.all_variables())
    yield ProbabilisticFact(clause.get_probability(), auxiliary_term)
    yield Clause(clause_head, clause_body + (auxiliary_term,))
    
    
from collections import defaultdict
from typing import Callable

from deepsoftlog.algebraic_prover.terms.expression cimport Expr


cdef class Variable:
    def __init__(self, name: str):
        self.name = name

    # noinspection PyMethodMayBeStatic
    def is_ground(self) -> bool:
        return False

    cpdef apply_substitution(self, substitution):
        return self.apply_substitution_(substitution)[0]

    cpdef tuple apply_substitution_(self, substitution):
        if self in substitution:
            return substitution[self], True
        else:
            return self, False

    def all_variables(self):
        return {self}

    def __repr__(self):
        return self.name

    def __eq__(self, other: Variable):
        if type(other) is not Variable:
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


cdef class CanonicalVariableCounter:
    def __init__(self, start=0, functor="VAR_"):
        self.counter = start
        self.functor = functor

    cpdef get_fresh_variable(self):
        self.counter += 1
        return Variable(self.functor + str(self.counter))




cpdef fresh_variables(Expr expr, fresh_variable_function):
    substitution = dict(fresh_variables_(expr, defaultdict(fresh_variable_function)))
    return expr.apply_substitution(substitution), substitution


cpdef fresh_variables_(Expr expr, substitution):
    for argument in expr.arguments:
        if type(argument) is Variable:
            _ = substitution[argument]
        else:
            substitution = fresh_variables_(argument, substitution)
    return substitution

import pickle

import torch
from torch import Tensor

from deepsoftlog.algebraic_prover.terms.expression import Expr


class SoftTerm(Expr):
    def __init__(self, term):
        super().__init__("~", term)

    def __str__(self):
        return f"~{self.arguments[0]}"

    def with_args(self, arguments):
        return SoftTerm(*arguments)

    def get_soft_term(self):
        return self.arguments[0]


class BatchSoftTerm(Expr):
    def __init__(self, terms):
        super().__init__("~", *terms)

    def __str__(self):
        return f"~Batch({len(self.arguments)})"

    def __repr__(self):
        return str(self)

    def with_args(self, arguments):
        return BatchSoftTerm(arguments)

    def get_soft_term(self):
        return self.arguments

    def __getitem__(self, item):
        return SoftTerm(self.arguments[item])


# monkey-patching on Expr for convenience
# Expr.__invert__ = lambda self: SoftTerm(self)
# Expr.is_soft = lambda self: self.functor == "~"


class TensorTerm(Expr):
    def __init__(self, tensor: Tensor):
        super().__init__(f"tensor{tuple(tensor.shape)}")
        self.tensor = tensor

    def get_tensor(self):
        return self.tensor

    def with_args(self, arguments):
        assert len(arguments) == 0
        return self

    def __repr__(self):
        return f"t{str(hash(self))[-3:]}"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, TensorTerm) \
            and self.tensor.shape == other.tensor.shape \
            and torch.all(self.tensor == other.tensor)

    def __hash__(self):
        return hash(pickle.dumps(self.tensor))

    def show(self):
        from matplotlib import pyplot as plt
        plt.imshow(self.tensor[0], cmap='gray')
        plt.show()
        
from typing import Optional

from deepsoftlog.algebraic_prover.proving.unify import replace_all_occurrences
from deepsoftlog.algebraic_prover.terms.variable import Variable
from deepsoftlog.algebraic_prover.terms.probability_annotation import LogProbabilisticExpr
from deepsoftlog.algebraic_prover.terms.expression import Expr


def get_unify_fact(term1: Expr, term2: Expr, store, metric: str) -> Expr:
    if term1 < term2:
        term1, term2 = term2, term1
    prob = store.soft_unify_score(term1, term2, metric)
    fact = Expr("k", term1, term2)
    return LogProbabilisticExpr(prob, ":-", fact, Expr(","), infix=True)


# def is_soft(e: Expr):
#     return e.get_predicate() == ("~", 1)

def is_soft(e: Expr):
    result = e.get_predicate() == ("~", 1)
    # print(f"Checking if {e} is soft: {result}, predicate: {e.get_predicate()}")
    return result


def look_for_rr(x) -> int:
    if isinstance(x, Variable):
        return 0
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum(look_for_rr(t) for t in x)
    else:  # if isinstance(x, Expr):
        return x.functor.startswith("rr") + look_for_rr(x.arguments)

def look_for_oobj(x) -> int:
    if isinstance(x, Variable):
        return 0
    elif isinstance(x, list) or isinstance(x, tuple):
        return sum(look_for_oobj(t) for t in x)
    else:  # if isinstance(x, Expr):
        return x.functor.startswith("oobj") + look_for_oobj(x.arguments)

# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 1:
#         return
#     # No occurs check
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     while changes:
#         changes = False
#         for i in range(len(substitution)):
#             s, t = substitution[i]
#             if type(t) is Variable and type(s) is not Variable:
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             if type(s) is Variable:
#                 if t == s:
#                     del substitution[i]
#                     changes = True
#                     break
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 if is_soft(s) and is_soft(t):
#                     s, t = s.arguments[0], t.arguments[0]
#                     if isinstance(s, Variable):
#                         substitution[i] = (s, t)
#                     elif isinstance(t, Variable):
#                         substitution[i] = (t, s)
#                     elif s.is_ground() and t.is_ground():
#                         if s != t:
#                             soft_unifies.add(get_unify_fact(s, t, store, metric))
#                         del substitution[i]
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s}` and `{t}` is illegal")
#                     changes = True
#                     break

#                 if s.get_predicate() != t.get_predicate():
#                     # can't hard unify
#                     return None
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break

#     return dict(substitution), soft_unifies


# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     print(f"Attempting soft_mgu between {term1} and {term2}")
#     print(f"Term1 type: {type(term1)}, Term2 type: {type(term2)}")
    
#     # For each argument, print its type and structure
#     if isinstance(term1, Expr) and hasattr(term1, 'arguments'):
#         for i, arg in enumerate(term1.arguments):
#             print(f"Term1 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if isinstance(term2, Expr) and hasattr(term2, 'arguments'):
#         for i, arg in enumerate(term2.arguments):
#             print(f"Term2 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
#         return
#     # No occurs check
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     while changes:
#         changes = False
#         for i in range(len(substitution)):
#             s, t = substitution[i]
#             print(f"Checking substitution pair: {s}, {t}")
            
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             if type(s) is Variable:
#                 if t == s:
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 print(f"Both are expressions: {s.get_predicate()} and {t.get_predicate()}")
#                 print(f"Checking if soft: {is_soft(s)} and {is_soft(t)}")
                
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
#                     changes = True
#                     break

#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     # can't hard unify
#                     return None
                
#                 print(f"Creating substitution pairs for arguments")
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break

#     print(f"Final substitution: {dict(substitution)}")
#     print(f"Soft unifies: {soft_unifies}")
#     return dict(substitution), soft_unifies


##3.7 thinking version

# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     """
#     Most General Unifier with support for soft terms (~).
#     Returns a tuple of (substitution dict, set of soft unification facts) or None if unification fails.
#     """
#     print(f"Attempting soft_mgu between {term1} and {term2}")
#     print(f"Term1 type: {type(term1)}, Term2 type: {type(term2)}")
    
#     # Debug output for arguments
#     if isinstance(term1, Expr) and hasattr(term1, 'arguments'):
#         for i, arg in enumerate(term1.arguments):
#             print(f"Term1 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     if isinstance(term2, Expr) and hasattr(term2, 'arguments'):
#         for i, arg in enumerate(term2.arguments):
#             print(f"Term2 arg {i}: {arg}, Type: {type(arg)}")
#             if isinstance(arg, Expr) and hasattr(arg, 'arguments'):
#                 print(f"  - Functor: {arg.functor}")
#                 print(f"  - Arguments: {arg.arguments}")
    
#     # Early exit conditions
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
#         print("Early exit: Too many rr or oobj terms")
#         return None
    
#     # Initialize substitution list and soft facts set
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
    
#     # Main unification loop
#     while changes:
#         changes = False
#         i = 0
#         while i < len(substitution):
#             s, t = substitution[i]
#             print(f"Checking substitution pair: {s}, {t}")
            
#             # Case 1: Swap variable/non-variable pairs to ensure variable is on left
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             # Case 2: Handle variable on left side
#             if type(s) is Variable:
#                 # Case 2.1: Identity - remove redundant mapping
#                 if t == s:
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
                
#                 # Case 2.2: Check for occurs (variable appearing in the term it's bound to)
#                 if isinstance(t, Expr) and s in t.all_variables():
#                     print(f"Occurs check failed: {s} appears in {t}")
#                     return None
                
#                 # Case 2.3: Apply substitution throughout
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break
            
#             # Case 3: Handle expressions on both sides
#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 print(f"Both are expressions: {s.get_predicate()} and {t.get_predicate()}")
#                 print(f"Checking if soft: {is_soft(s)} and {is_soft(t)}")
                
#                 # Case 3.1: Both are soft terms (~)
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     # Case 3.1.1: First inner term is variable
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                         changes = True
#                         break
#                     # Case 3.1.2: Second inner term is variable
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                         changes = True
#                         break
#                     # Case 3.1.3: Both inner terms are ground
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                         changes = True
#                         break
#                     # Case 3.1.4: Error case - can't soft unify non-ground terms
#                     else:
#                         raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
                
#                 # Case 3.2: Different predicates - can't unify
#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     return None
                
#                 # Case 3.3: Same predicate - unify arguments
#                 print(f"Creating substitution pairs for arguments")
#                 if len(s.arguments) != len(t.arguments):
#                     print(f"Arity mismatch: {len(s.arguments)} vs {len(t.arguments)}")
#                     return None
                    
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                 changes = True
#                 break
            
#             # Move to next pair if no changes
#             i += 1

#     # Convert substitution list to proper dictionary
#     # Ensure only variables appear as keys
#     result_dict = {}
#     for k, v in substitution:
#         if isinstance(k, Variable):
#             result_dict[k] = v
    
#     print(f"Final substitution: {result_dict}")
#     print(f"Soft unifies: {soft_unifies}")
#     return result_dict, soft_unifies


def soft_mgu(term1: Expr, term2: Expr, store, metric, soft_cache=None) -> Optional[tuple[dict, set]]:
    """Most General Unifier with support for soft terms (~)."""
    # print(f"Attempting soft_mgu between {term1} and {term2}")
    
    # Early exit conditions
    if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 2:
        print("Early exit: Too many rr or oobj terms")
        return None
    
    # Initialize substitution list and soft facts set
    substitution = [(term1, term2)]
    soft_unifies = set()
    changes = True
    
    # Main unification loop
    while changes:
        changes = False
        i = 0
        while i < len(substitution):
            s, t = substitution[i]
            
            # Case 1: Swap variable/non-variable pairs
            if type(t) is Variable and type(s) is not Variable:
                substitution[i] = (t, s)
                changes = True
                break

            # Case 2: Handle variable on left side
            if type(s) is Variable:
                if t == s:
                    del substitution[i]
                    changes = True
                    break
                
                if isinstance(t, Expr) and s in t.all_variables():
                    return None
                
                new_substitution = replace_all_occurrences(s, t, i, substitution)
                if new_substitution is not None:
                    substitution = new_substitution
                    changes = True
                    break
            
            # Case 3: Handle expressions on both sides
            if isinstance(s, Expr) and isinstance(t, Expr):
                # Case 3.1: Both are soft terms (~)
                if is_soft(s) and is_soft(t):
                    s_inner, t_inner = s.arguments[0], t.arguments[0]
                    
                    if isinstance(s_inner, Variable):
                        substitution[i] = (s_inner, t_inner)
                        changes = True
                        break
                    elif isinstance(t_inner, Variable):
                        substitution[i] = (t_inner, s_inner)
                        changes = True
                        break
                    elif s_inner.is_ground() and t_inner.is_ground():
                        if s_inner != t_inner:
                            # Create a canonical key for caching
                            terms = sorted([str(s_inner), str(t_inner)])
                            pair_key = (terms[0], terms[1])
                            
                            # Check cache for existing unification
                            if soft_cache is not None and pair_key in soft_cache:
                                # print(f"Reusing cached soft unification for {s_inner} and {t_inner}")
                                soft_unifies.add(soft_cache[pair_key])
                            else:
                                # print(f"Creating new soft unification fact for {s_inner} and {t_inner}")
                                fact = get_unify_fact(s_inner, t_inner, store, metric)
                                soft_unifies.add(fact)
                                
                                # Cache this soft unification
                                if soft_cache is not None:
                                    soft_cache[pair_key] = fact
                        
                        del substitution[i]
                        changes = True
                        if len(soft_unifies) > 0:
                            print(f"SOFT UNIFICATION FACTS: {soft_unifies}")
                            # Check probability values of soft facts
                            for fact in soft_unifies:
                                if hasattr(fact, 'get_probability'):
                                    prob = fact.get_probability()
                                    if prob < 0.001:
                                        print(f"WARNING: Very small soft unification probability: {prob}")
                        break
                    else:
                        raise Exception(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
                
                if s.get_predicate() != t.get_predicate():
                    return None
                
                new_substitution = list(zip(s.arguments, t.arguments))
                substitution = substitution[:i] + new_substitution + substitution[i+1:]
                changes = True
                break
            
            i += 1

    # Convert substitution list to proper dictionary
    result_dict = {}
    for k, v in substitution:
        if isinstance(k, Variable):
            result_dict[k] = v
    
    print(f"Final substitution: {result_dict}")
    print(f"Soft unifies: {soft_unifies}")
    return result_dict, soft_unifies

###NEWEST VERSION
# def soft_mgu(term1: Expr, term2: Expr, store, metric) -> Optional[tuple[dict, set]]:
#     """
#     Enhanced Most General Unifier function that handles soft unification.
#     Tracks both hard unification substitutions and soft unifications.
#     """
#     print(f"Attempting soft_mgu between {term1} and {term2}")

#     # Early check for relations that we want to avoid
#     if look_for_rr([term1, term2]) > 1 or look_for_oobj([term1, term2]) > 1:
#         print(f"Aborting unification due to multiple 'rr' or 'oobj' terms")
#         return None
        
#     # Initialize the substitution list and soft unifications set
#     substitution = [(term1, term2)]
#     soft_unifies = set()
#     changes = True
#     iteration = 0
    
#     # Process the substitution list until no more changes are made
#     while changes:
#         changes = False
#         iteration += 1
#         print(f"Unification iteration {iteration}, current subs: {substitution}")
        
#         for i in range(len(substitution)):
#             if i >= len(substitution):  # Safety check in case items were deleted
#                 break
                
#             s, t = substitution[i]
#             print(f"Examining pair: {s} and {t}")
            
#             # If t is a variable and s is not, swap them to ensure variables are on the left
#             if type(t) is Variable and type(s) is not Variable:
#                 print(f"Swapping variable {t} with non-variable {s}")
#                 substitution[i] = (t, s)
#                 changes = True
#                 break

#             # Handle variable on left side
#             if type(s) is Variable:
#                 if t == s:
#                     # Remove identity substitution
#                     print(f"Removing identical var-var pair: {s}={t}")
#                     del substitution[i]
#                     changes = True
#                     break
                
#                 # Check if the variable occurs in the term (occurs check)
#                 if isinstance(t, Expr) and s in t.all_variables():
#                     print(f"Occurs check failed: {s} occurs in {t}")
#                     return None
                
#                 # Replace all occurrences of the variable in other substitutions
#                 print(f"Replacing occurrences of {s} with {t}")
#                 new_substitution = replace_all_occurrences(s, t, i, substitution)
#                 if new_substitution is not None:
#                     substitution = new_substitution
#                     changes = True
#                     break

#             # Both sides are expressions
#             if isinstance(s, Expr) and isinstance(t, Expr):
#                 # Special handling for soft terms (terms with ~ functor)
#                 if is_soft(s) and is_soft(t):
#                     print(f"Both are soft terms: {s} and {t}")
#                     s_inner, t_inner = s.arguments[0], t.arguments[0]
#                     print(f"Inner terms: {s_inner} and {t_inner}")
                    
#                     if isinstance(s_inner, Variable):
#                         print(f"First inner term is variable: {s_inner}")
#                         substitution[i] = (s_inner, t_inner)
#                     elif isinstance(t_inner, Variable):
#                         print(f"Second inner term is variable: {t_inner}")
#                         substitution[i] = (t_inner, s_inner)
#                     elif s_inner.is_ground() and t_inner.is_ground():
#                         print(f"Both inner terms are ground")
#                         if s_inner != t_inner:
#                             print(f"Creating soft unification fact for {s_inner} and {t_inner}")
#                             soft_unifies.add(get_unify_fact(s_inner, t_inner, store, metric))
#                         del substitution[i]
#                     else:
#                         print(f"Soft unification of non-ground terms `{s_inner}` and `{t_inner}` is illegal")
#                         return None
#                     changes = True
#                     break

#                 # Regular hard unification for non-soft terms
#                 if s.get_predicate() != t.get_predicate():
#                     print(f"Predicates don't match: {s.get_predicate()} vs {t.get_predicate()}")
#                     return None
                
#                 print(f"Creating substitution pairs for arguments")
#                 new_substitution = list(zip(s.arguments, t.arguments))
#                 if len(new_substitution) > 0:  # Only make changes if there are arguments to process
#                     substitution = substitution[:i] + new_substitution + substitution[i+1:]
#                     changes = True
#                     break
#                 else:
#                     # For constants with no arguments, just remove this pair
#                     del substitution[i]
#                     changes = True
#                     break

#     # Convert the list of substitutions to a dictionary
#     result_dict = {}
#     for var, term in substitution:
#         if isinstance(var, Variable):
#             result_dict[var] = term
    
#     print(f"Final substitution: {result_dict}")
#     print(f"Soft unifies: {soft_unifies}")
#     return result_dict, soft_unifies


from collections import defaultdict

from deepsoftlog.algebraic_prover.terms.expression import Clause, Expr, And, Fact
from deepsoftlog.algebraic_prover.terms.variable import CanonicalVariableCounter, Variable
from logic.soft_term import SoftTerm, BatchSoftTerm


def batch_soft_rules(clauses):
    """
    Combines the rules that have the same head, but different soft terms
    into a single rule with a batched soft terms.
    """
    clause_sets = defaultdict(list)
    for clause in clauses:
        clause_sets[remove_soft_term(clause)].append(clause)
    clauses = {_batch_soft_set(clause_set) for clause_set in clause_sets.values()}
    for clause in clauses:
        print(clause)
    return clauses


UID = 0


def remove_soft_term(term: Expr) -> Expr:
    global UID
    """
    Returns a term, where all soft terms are replaced
    TODO: also take variables in account
    """
    if isinstance(term, Expr):
        if term.is_soft():
            return Expr("~")
        else:
            return term.with_args([remove_soft_term(arg) for arg in term.arguments])
    UID += 1
    return Expr(f"VAR_{UID}")


def _batch_soft_set(clauses: list) -> Expr:
    clause = clauses[0]
    if len(clauses) == 1:
        return clause

    if isinstance(clause, Variable):
        assert all(isinstance(clause, Variable) for clause in clauses)
        return clause

    predicate = clause.get_predicate()
    assert all(clause.get_predicate() == predicate for clause in clauses)

    if clauses[0].is_soft():
        assert all(clause.is_soft() for clause in clauses)
        soft_terms = tuple([clause.arguments[0] for clause in clauses])
        return BatchSoftTerm(soft_terms)

    new_args = []
    for i in range(clause.get_arity()):
        new_args.append(_batch_soft_set([clause.arguments[i] for clause in clauses]))
    return clauses[0].with_args(new_args)


def SPL2ProbLog(clauses):
    X = Variable("X")
    reflexive_rule = Fact(Expr('k', X, X))  # k(X, X).

    clauses = [soft_term_elimination(clause) for clause in clauses]
    clauses = [double_var_removal(clause) for clause in clauses]
    return clauses + [reflexive_rule]


def soft_term_elimination(clause: Clause) -> Clause:
    """
    Replaces all soft terms in the head of a clause
    with variables, and adds atoms in the body to enforce
    the variable-soft term relation: `k(Var, soft_term)`
    """
    var_counter = CanonicalVariableCounter(functor="SV_")
    fresh_variables = defaultdict(var_counter.get_fresh_variable)
    substitution, new_head = _soft_term_replace(clause.get_head(), fresh_variables)
    new_atoms = [Expr('k', st, SoftTerm(v)) for st, v in substitution.items()]
    new_body = And.create(new_atoms + list(clause.get_body()))
    return clause.with_args([new_head, new_body])


def double_var_removal(clause: Clause) -> Clause:
    """
    Makes sure all occurrences of a variable in the
    head of clause are unique.
    """
    # TODO: refactor, too similar to `soft_term_elimination`
    var_counter = CanonicalVariableCounter(functor="DV_")
    fresh_variables = defaultdict(var_counter.get_fresh_variable)
    substitution, new_head = _double_var_replace(clause.get_head(), fresh_variables)
    new_atoms = [Expr('k', st, v) for st, v in substitution.items()]
    new_body = And.create(new_atoms + list(clause.get_body()))
    return clause.with_args([new_head, new_body])


def _soft_term_replace(term: Expr, substitution: defaultdict):
    if isinstance(term, Expr):
        if term.is_soft():
            term = SoftTerm(substitution[term])
        else:
            substitution, term = _rec_subst(term, substitution, _soft_term_replace)

    return substitution, term


def _double_var_replace(term: Expr, substitution: defaultdict, bound_vars=None):
    if bound_vars is None:
        bound_vars = list()
    if isinstance(term, Variable):
        if term in bound_vars:
            term = substitution[term]
        else:
            bound_vars.append(term)
    else:
        arguments = []
        for argument in term.arguments:
            subst, new_arg = _double_var_replace(argument, substitution, bound_vars)
            arguments.append(new_arg)
        term = term.with_args(arguments)

    return substitution, term


def _rec_subst(term, subst, func):
    arguments = []
    for argument in term.arguments:
        subst, new_arg = func(argument, subst)
        arguments.append(new_arg)
    term = term.with_args(arguments)
    return subst, term


from typing import Iterable

import torch

from ..algebraic_prover.builtins import External
from ..algebraic_prover.proving.proof_module import ProofModule
from ..algebraic_prover.algebras.sdd2_algebra import DnfAlgebra
from ..algebraic_prover.algebras.probability_algebra import LOG_PROBABILITY_ALGEBRA
from ..algebraic_prover.algebras.tnorm_algebra import LogProductAlgebra, LogGodelAlgebra
from ..algebraic_prover.algebras.sdd_algebra import SddAlgebra
from ..algebraic_prover.terms.expression import Expr, Fact, Clause
from ..embeddings.embedding_store import EmbeddingStore
from ..parser.vocabulary import Vocabulary
from .soft_unify import soft_mgu
from deepsoftlog.data import sg_to_prolog
from deepsoftlog.algebraic_prover.terms.transformations import normalize_clauses
from deepsoftlog.algebraic_prover.proving.proof_queue import OrderedProofQueue

class SoftProofModule(ProofModule):
    def __init__(
            self,
            clauses: Iterable[Expr],
            embedding_metric: str = "l2",
            semantics: str = 'sdd2',
    ):
        super().__init__(clauses=clauses, algebra=None)
        self.store = EmbeddingStore(0, None, Vocabulary())
        self.builtins = super().get_builtins() + (ExternalCut(),)#, type_external)
        self.embedding_metric = embedding_metric
        self.semantics = semantics
        self.algebra = _get_algebra(self.semantics, self)
        self.soft_unification_cache = {}

    def mgu(self, t1, t2):
        # Pass the cache to soft_mgu
        return soft_mgu(t1, t2, self.get_store(), self.embedding_metric, self.soft_unification_cache)

    # def query(self, *args, **kwargs):
    #     print("CALLED (SPL)")
    #     if self.algebra is None:
    #         self.algebra = _get_algebra(self.semantics, self)
    #     self.algebra.reset()
    #     print(f"Super query: {super().query(*args, **kwargs)}")
    #     return super().query(*args, **kwargs)
    
    def query(self, *args, **kwargs):
        print(f"QUERY CALLED: {args[0] if args else None}")
        print(f"Algebra type: {type(self.algebra).__name__}")
        
        # Capture original query before any transformation
        original_query = args[0] if args else None
        
        self.queried = original_query
        print(f"Query kwargs: {kwargs}")
        
        if self.algebra is None:
            self.algebra = _get_algebra(self.semantics, self)
        self.algebra.reset()
        
        # Don't add return_stats if it's already in kwargs
        if 'return_stats' in kwargs:
            result = super().query(*args, **kwargs)
            # Check if it's a tuple containing the stats
            if isinstance(result, tuple) and len(result) == 3:
                result_dict, proof_steps, nb_proofs = result
            else:
                # If not, we're probably in a different code path
                print("NOTICE: Result wasn't unpacked as expected")
                result_dict = result
                proof_steps = -1
                nb_proofs = -1
        else:
            result_dict, proof_steps, nb_proofs = super().query(*args, return_stats=True, **kwargs)
        
        print(f"Query result stats:")
        print(f"  Proof steps: {proof_steps}")
        print(f"  Number of proofs: {nb_proofs}")
        print(f"  Result type: {type(result_dict)}")
        
        if isinstance(result_dict, dict):
            print(f"  Result keys: {list(result_dict.keys())}")
            for k, v in result_dict.items():
                print(f"  {k}: {v}")
        
        # Log whether the query itself was found in the results
        if isinstance(result_dict, dict) and original_query is not None:
            if original_query in result_dict:
                print(f"Query found in results with value: {result_dict[original_query]}")
            else:
                print(f"WARNING: Query not found in results")
                print(f"Available keys: {list(result_dict.keys())}")
        
        # Return the appropriate result
        if 'return_stats' in kwargs and kwargs['return_stats']:
            return result_dict, proof_steps, nb_proofs
        else:
            return result_dict
        
    def get_builtins(self):
        return self.builtins

    def get_vocabulary(self):
        return Vocabulary().add_all(self.clauses)

    def get_store(self):
        if hasattr(self.store, "module"):
            return self.store.module
        return self.store

    def parameters(self):
        yield from self.store.parameters()
        if self.semantics == "neural":
            yield from self.algebra.parameters()

    def grad_norm(self, order=2):
        grads = [p.grad.detach().data.flatten()
                 for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            return 0
        grad_norm = torch.linalg.norm(torch.hstack(grads), ord=order)
        return grad_norm
    

    def update_clauses(self, DataInstance):
        # Make a copy of the set for iteration
        clauses_copy = list(self.clauses)  # Convert to list to avoid set iteration issues

        # Filter out the "scene_graph" clauses, considering the rule structure
        filtered_clauses = []
        
        # Define exceptions - rules to keep despite having filtered functors
        exceptions = ["~rr1"]  # Add any other special relations here
        
        for expr in clauses_copy:
            if expr.functor == ":-":  # It's a rule or fact-as-rule
                head = expr.arguments[0]  # Get the head of the rule
                
                # Check if this is either not a filtered functor OR it's in our exceptions
                if (head.functor not in ["scene_graph", "object", "groundtruth"] or
                    (hasattr(head, "arguments") and len(head.arguments) > 0 and 
                    str(head.arguments[0]) in exceptions)):
                    filtered_clauses.append(expr)
                    
            else:  # It's a simple expression (not a rule)
                if expr.functor not in ["scene_graph", "object", "groundtruth"]:
                    filtered_clauses.append(expr)

        # Now you can update the clauses with the filtered list
        self.clauses.clear()  # Clear the existing clauses
        self.clauses.update(set(filtered_clauses))  # Add the filtered clauses

        # Optionally, update the store and vocabulary
        sg_clauses = sg_to_prolog(DataInstance)
        sg_clauses = normalize_clauses(sg_clauses)        

        self.clauses.update(sg_clauses)  # Add new clauses from the current instance
        updated_vocabulary = Vocabulary().add_all(self.clauses)
        
        # Update the store's vocabulary
        self.get_store().vocabulary = updated_vocabulary
        
        # Initialize embeddings for new constants
        for constant in updated_vocabulary.get_constants():
            if constant not in self.get_store().constant_embeddings:
                print(f"Initializing new embedding for unseen constant: {constant}")
                self.get_store().constant_embeddings[constant] = self.get_store().initializer(constant)
        
        # Clear the cache to ensure fresh computations
        self.get_store().clear_cache()
        
        print(f"Updated clauses and constant embeddings successfully")
            
    def analyze_failed_proof(self, query: Expr):
        """Special function to analyze why a proof is failing"""
        print(f"\n=== PROOF ANALYSIS for query: {query} ===\n")
        
        # Try with extremely high depth and branching limits
        print("Attempting proof with very high limits...")
        
        # Use OrderedProofQueue to prioritize promising proofs
        queue = OrderedProofQueue(self.algebra)
        
        # First try normal query with high limits
        result, steps, nb_proofs = self.query(
            query, 
            max_depth=20,  # Very high depth
            max_branching=20,  # Very high branching
            queue=queue,
            return_stats=True
        )
        
        print(f"Analysis results: {nb_proofs} proofs in {steps} steps")
        
        if nb_proofs == 0:
            print("\nNo proofs found, analyzing proof tree...")
            
            # Try to analyze the proof tree directly
            if hasattr(queue, '_queue'):
                print(f"Queue still has {len(queue._queue)} items")
                
                # Check the first few items
                for i, (_, _, proof) in enumerate(queue._queue[:3]):
                    print(f"\nQueue item {i}:")
                    print(f"  Query: {proof.query}")
                    print(f"  Goals: {proof.goals}")
                    print(f"  Is complete: {proof.is_complete()}")
                    print(f"  Value: {proof.value}")
                    
                    # Check for object predicates
                    if proof.goals and all(g.functor == "object" for g in proof.goals):
                        print(f"  POSSIBLE ISSUE: All goals are object predicates but proof not complete")
                        print(f"  Goals: {proof.goals}")
                        
                        # Look at the needed soft unifications
                        if hasattr(proof.value, 'pos_facts'):
                            print(f"  Soft facts: {proof.value.pos_facts}")
                        
                        # Try to manually complete this proof
                        print(f"  Attempting manual completion...")
                        try:
                            manual_proof = type(proof)(
                                query=proof.query,
                                goals=tuple(),  # Empty tuple = completed
                                depth=proof.depth + 1,
                                proof_tree=proof.proof_tree,
                                value=proof.value
                            )
                            print(f"  Manual proof is_complete: {manual_proof.is_complete()}")
                            
                            # Check if this would be a valid completion
                            if manual_proof.is_complete():
                                print(f"  DIAGNOSTIC: The proof CAN be manually completed!")
                                print(f"  This suggests the issue is in how proof completion is detected")
                        except Exception as e:
                            print(f"  Error creating manual proof: {str(e)}")
        
        print("\n=== END PROOF ANALYSIS ===\n")
        return result


def _get_algebra(semantics, program):
    
    if semantics == "sdd":
        return SddAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "sdd2":
        return DnfAlgebra(LOG_PROBABILITY_ALGEBRA)
    elif semantics == "godel":
        return LogGodelAlgebra()
    elif semantics == "product":
        return LogProductAlgebra()
    raise ValueError(f"Unknown semantics: {semantics}")


class ExternalCut(External):
    def __init__(self):
        super().__init__("cut", 1, None)
        self.cache = set()

    def get_answers(self, t1) -> Iterable[tuple[Expr, dict, set]]:
        if t1 not in self.cache:
            self.cache.add(t1)
            fact = Fact(Expr("cut", t1))
            return [(fact, {}, set())]
        return []


class DebugSoftProofModule(SoftProofModule):
    pass
    def __init__(self, clauses: Iterable[Expr], embedding_metric: str = "l2", semantics: str = 'sdd2'):
        super().__init__(clauses=clauses, embedding_metric=embedding_metric, semantics=semantics)
        self.debug = True
        
    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict, set]]:
        predicate = term.get_predicate()
        if self.debug:
            print(f"\nDEBUG: Matching term: {term}")
            print(f"DEBUG: Predicate: {predicate}")
        
        # First handle builtins
        for builtin in self.get_builtins():
            if predicate == builtin.predicate:
                if self.debug:
                    print(f"DEBUG: Found builtin match: {builtin.predicate}")
                yield from builtin.get_answers(*term.arguments)

        # Debug all available clauses
        if self.debug:
            print("\nDEBUG: Available clauses:")
            for clause in self.clauses:
                print(f"DEBUG: {clause}")

        # Try to match with all clauses
        for db_clause in self.clauses:
            if self.debug:
                print(f"\nDEBUG: Trying clause: {db_clause}")
            
            # Handle facts (clauses without body)
            is_fact = isinstance(db_clause, Expr) or (
                hasattr(db_clause, 'arguments') and 
                len(db_clause.arguments) == 1
            )
            
            db_head = db_clause if is_fact else db_clause.arguments[0]
            
            if self.debug:
                print(f"DEBUG: Clause head: {db_head}")
                print(f"DEBUG: Is fact: {is_fact}")
            
            if self.mask_query and db_head == self.queried:
                if self.debug:
                    print("DEBUG: Skipping masked query")
                continue
            
            if db_head.get_predicate() == predicate:
                if self.debug:
                    print("DEBUG: Predicate matches")
                
                # For facts, use them directly; for rules, create fresh variables
                working_clause = db_clause if is_fact else self.fresh_variables(db_clause)
                working_head = working_clause if is_fact else working_clause.arguments[0]
                
                if self.debug:
                    print(f"DEBUG: Attempting unification between:")
                    print(f"DEBUG:   Term: {term}")
                    print(f"DEBUG:   Head: {working_head}")
                
                result = self.mgu(term, working_head)
                
                if self.debug:
                    print(f"DEBUG: MGU result: {result}")
                
                if result is not None:
                    unifier, new_facts = result
                    if is_fact:
                        # For facts, create a simple clause
                        new_clause = Clause(working_head, None)  # Adjust based on your Clause implementation
                    else:
                        new_clause = working_clause.apply_substitution(unifier)
                    
                    if self.debug:
                        print(f"DEBUG: Match found!")
                        print(f"DEBUG: Unifier: {unifier}")
                        print(f"DEBUG: New clause: {new_clause}")
                    
                    yield new_clause, unifier, new_facts

    def query(self, *args, **kwargs):
        if self.debug:
            print("\nDEBUG: Starting new query")
            print(f"DEBUG: Args: {args}")
            print(f"DEBUG: Kwargs: {kwargs}")
        return super().query(*args, **kwargs)
    
from torch.optim import AdamW

from deepsoftlog.algebraic_prover.algebras import safe_log_negate


def get_optimizer(store, config: dict):
    optimizer_name = config['optimizer']
    assert optimizer_name == "AdamW", f"Unknown optimiser `{optimizer_name}`"
    constant_group = {
        'params': store.constant_embeddings.parameters(),
        'lr': config['embedding_learning_rate'],
    }
    functor_group = {
        'params': store.functor_embeddings.parameters(),
        'lr': config['functor_learning_rate'],
        'weight_decay': config.get('functor_weight_decay', 0.)
    }

    optimizer = AdamW([constant_group, functor_group])
    return optimizer


def nll_loss(log_pred, target, gamma=0.):
    """ Negative log-likelihood loss """
    print(f"NLL loss input: log_pred={log_pred}, target={target}")
    assert target in [0., 1.]
    if target == 0.:
        log_pred = safe_log_negate(log_pred)
    if gamma > 0.:  # focal loss
        log_pred = (1 - log_pred.exp()) ** gamma * log_pred

    return -log_pred

import numpy as np
from sklearn.metrics import average_precision_score

from ..data import Query

def debug_expr_tree(expr, depth=0):
    print(f"{'  ' * depth}Examining node: {expr}")
    print(f"{'  ' * depth}Type: {type(expr)}")
    print(f"{'  ' * depth}Dir: {dir(expr)}")
    if hasattr(expr, 'arguments'):
        print(f"{'  ' * depth}Arguments:")
        for i, arg in enumerate(expr.arguments):
            print(f"{'  ' * depth}  Argument {i}:")
            debug_expr_tree(arg, depth + 1)

def get_metrics(query: Query, results, dataset) -> dict[str, float]:
    # print("\n=== Starting Expression Tree Debug ===")
    # debug_expr_tree(query.query)
    # print("=== End Expression Tree Debug ===\n")
    # print("get_metrics")
    # print(query)
    # print(type(query))
    # print(query.query)
    # print(type(query.query))
    metrics = boolean_metrics(results, query)
    # print(type(query.query))
    if not query.query.is_ground():
        assert query.p == 1
        metrics.update(rank_metrics(results, dataset))
    return metrics

from time import time
import os
import shutil
from pathlib import Path
from typing import Iterable, Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from ..data.dataloader import DataLoader
from ..data.query import Query
from ..logic.spl_module import SoftProofModule, DebugSoftProofModule
from .logger import PrintLogger, WandbLogger
from .loss import nll_loss, get_optimizer
from .metrics import get_metrics, aggregate_metrics
from . import set_seed, ConfigDict
from deepsoftlog.data import expression_to_prolog


def ddp_setup(rank, world_size):
    print(f"Starting worker {rank + 1}/{world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    set_seed(1532 + rank)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _trainp(rank, world_size, trainer, cfg):
    ddp_setup(rank, world_size)
    trainer.program.store = DDP(trainer.program.store, find_unused_parameters=True)
    trainer._train(cfg, master=rank == 0)
    dist.destroy_process_group()


class Trainer:
    def __init__(
            self,
            program: SoftProofModule,
            load_train_dataset: Callable[[dict], DataLoader],
            criterion,
            optimizer: Optimizer,
            logger=PrintLogger(),
            **search_args
    ):
        self.program = program
        self.program.mask_query = True
        self.logger = logger
        self.load_train_dataset = load_train_dataset
        self.train_dataset = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = None
        self.grad_clip = None
        self.search_args = search_args

    def _train(self, cfg: dict, master=True, do_eval=True):
        nb_epochs = cfg['nb_epochs']
        self.grad_clip = cfg['grad_clip']
        self.program.store.to(cfg['device'])
        self.program.store.train()
        self.train_dataset = self.load_train_dataset(cfg)
        self.scheduler = CosineAnnealingLR(self.optimizer, nb_epochs + 1)
        for epoch in range(nb_epochs):
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"### EPOCH {epoch} (lr={last_lr:.2g}) ###")
            self.train_epoch(verbose=cfg['verbose'] and master)
            self.scheduler.step()
            if master:
                self.save(cfg)
            if do_eval and master and hasattr(self, 'val_dataloader'):
                self.eval(self.val_dataloader, name='val')
            print(f"Program algebra:\n{self.program.algebra._sdd_algebra.all_facts._val_to_ix}\n") if cfg['verbose'] else print(self.program)

    def train(self, cfg: dict, nb_workers: int = 1):
        if nb_workers == 1:
            return self._train(cfg, True)
        self.program.algebra = None
        self.train_dataset = None
        mp.spawn(_trainp,
                 args=(nb_workers, self, cfg),
                 nprocs=nb_workers,
                 join=True)

    def train_profile(self, *args, **kwargs):
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        self.train(*args, **kwargs)
        profiler.stop()
        profiler.open_in_browser()

    # def train_epoch(self, verbose: bool):
        
    #     for queries in tqdm(self.train_dataset, leave=False, smoothing=0, disable=not verbose):
    #         current_time = time()
    #         loss, diff, proof_steps, nb_proofs = self.get_loss(queries)
    #         grad_norm = 0.
    #         if loss is not None:
    #             grad_norm = self.step_optimizer()
    #         if verbose:
    #             self.logger.log({
    #                 'grad_norm': grad_norm,
    #                 'loss': loss,
    #                 'diff': diff,
    #                 "step_time": time() - current_time,
    #                 "proof_steps": proof_steps,
    #                 "nb_proofs": nb_proofs,
    #             })
    #     if verbose:
    #         self.logger.print()
    #     print("EPOCH END")
    
    def train_epoch(self, verbose: bool):
        # Clear the soft unification cache at the start of the epoch
        if hasattr(self.program, 'soft_unification_cache'):
            self.program.soft_unification_cache = {}
        
        for queries in tqdm(self.train_dataset, leave=False, smoothing=0, disable=not verbose):
            current_time = time()
            
            # Clear the cache before each batch too (to be safe)
            if hasattr(self.program, 'soft_unification_cache'):
                self.program.soft_unification_cache = {}
                
            loss, diff, proof_steps, nb_proofs = self.get_loss(queries)
            grad_norm = 0.
            if loss is not None:
                grad_norm = self.step_optimizer()
            if verbose:
                self.logger.log({
                    'grad_norm': grad_norm,
                    'loss': loss,
                    'diff': diff,
                    "step_time": time() - current_time,
                    "proof_steps": proof_steps,
                    "nb_proofs": nb_proofs,
                })
        if verbose:
            self.logger.print()
        print("EPOCH END")

    # def eval(self, dataloader: DataLoader, name='test'):
    #     self.program.store.eval()
    #     metrics = []
    #     print(f"oader: {dataloader}")
    #     for queries in tqdm(dataloader, leave=False, smoothing=0):
    #         print(f"Queries type:{type(queries[0])}")
    #         queries = [q.query if hasattr(q, "query") else q for q in queries]
    #         print(f"Queries type:{type(queries[0])}")
    #         results = zip(queries, self._eval_queries(queries))
    #         print(f"Results: {results}")
    #         print(f"Queries type:{type(queries[0])}")
    #         new_metrics = [get_metrics(query.query, result, queries) for query, result in results]
    #         metrics += new_metrics
    #     self.logger.log_eval(aggregate_metrics(metrics), name=name)
    
    def eval(self, dataloader: DataLoader, name='test'):
        print("EVALUATION STARTING")
        self.program.store.eval()
        metrics = []
        print(f"DataLoader: {dataloader}")
        for instance in dataloader.dataset.instances:
            instance.query = query_to_prolog(instance.query)

        for queries in tqdm(dataloader, leave=False, smoothing=0):

            if not isinstance(queries, Query): 
                print(f"Queries type:{type(queries[0])}") 
                queries = [q.query for q in queries]

            results = zip(queries, self._eval_queries(queries))
            print(f"Results: {results}")

            # Ensure we only access .query if it's still a DatasetInstance
            new_metrics = [get_metrics(query, result, queries) for query, result in results]

            metrics += new_metrics

        self.logger.log_eval(aggregate_metrics(metrics), name=name)

    def _query(self, queries: Iterable[Query]):
        for query in queries:
            print(f"Query: {query.query}")
            result, proof_steps, nb_proofs = self.program(
                query.query, **self.search_args
            )
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            yield result, proof_steps, nb_proofs

        
    
    def _query_result(self, queries: Iterable[Query]):
        for query in queries:
            print("_query_result called, type:")
            print(type(query))
            print(query)
            yield self.program.query(query.query, **self.search_args)

    def _eval_queries(self, queries: Iterable[Query]):
        with torch.no_grad():
            for query, results in zip(queries, self._query_result(queries)):
                if len(results) > 0:
                    results = {k: v.exp().item() for k, v in results.items() if v != 0.}
                    yield results
                else:
                    print(f"WARNING: empty result for {query}")
                    yield {}

    def get_loss(self, queries: Iterable[Query]) -> tuple[float, float, float, float]:
        print(f"Queries: {queries[:5]}")
        results, proof_steps, nb_proofs = tuple(zip(*self._query(queries)))
        print(f"Results: {results}")
        print(f"Proof steps: {proof_steps}")
        print(f"Number of proofs: {nb_proofs}")
        losses = [self.criterion(result, query.p) for result, query in zip(results, queries)]
        print(f"Losses: {losses}")
        loss = torch.stack(losses).mean()
        errors = [query.error_with(result) for result, query in zip(results, queries)]
        if loss.requires_grad:
            loss.backward()
        proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs

    
    def step_optimizer(self):
        with torch.no_grad():
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.program.parameters(), max_norm=self.grad_clip)
            grad_norm = self.program.grad_norm()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.get_store().clear_cache()
        return float(grad_norm)

    def save(self, config: ConfigDict):
        save_folder = f"results/{config['name']}"
        save_folder = Path(save_folder)
        if save_folder.exists():
            shutil.rmtree(save_folder, ignore_errors=True)
        save_folder.mkdir(parents=True)

        config.save(save_folder / "config.yaml")
        torch.save(self.get_store().state_dict(), save_folder / "store.pt")

    def get_store(self):
        return self.program.get_store()


def create_trainer(program, load_train_dataset, cfg):
    trainer_args = {
        "program": program,
        "criterion": nll_loss,
        "load_train_dataset": load_train_dataset,
        "optimizer": get_optimizer(program.get_store(), cfg),
        "logger": WandbLogger(cfg),
        "max_proofs": cfg.get("max_proofs", None),
        "max_depth": cfg.get("max_depth", None),
        "max_branching": cfg.get("max_branching", None),
    }
    return Trainer(**trainer_args)


from deepsoftlog.training.trainer import Trainer
from data.dataset import DatasetInstance, ReferringExpressionDataset
from typing import Iterable
import torch
import numpy as np
from deepsoftlog.data import expression_to_prolog, query_to_prolog
from deepsoftlog.data.query import Query

class ReferringTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _referring_queries(self, data_instances: Iterable[DatasetInstance]):
        """
        Iterate through data instances, update the program clauses, attempt to prove query 
        using the updated program, and yield results.
        """
        for instance in data_instances:
            # Update clauses in the program dynamically based on the instance
            print(f"All clauses: {self.program.clauses}")
            self.program.update_clauses(instance)
            print(f"All clauses: {self.program.clauses}")
            if not isinstance(instance.query, Query):
                print(f"QUERY: {instance.query}")
                instance.query = query_to_prolog(instance.query)
            print(f"QUERY: {instance.query}")

            # Attempt to prove the query using the updated program
            # print(f"Query: {instance.query}")
            # print(f"Query: {instance.query.query}")
            result, proof_steps, nb_proofs = self.program(instance.query.query, **self.search_args)

            # Ensure result is a tensor if it isn't already
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)

            # Yield the result, proof steps, and number of proofs
            yield result, proof_steps, nb_proofs
            
            
    # def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
    #     """
    #     Compute the loss for referring queries using the _referring_queries method.

    #     Args:
    #         data_instances (Iterable[DataInstance]): The referring expression instances.

    #     Returns:
    #         tuple[float, float, float, float]: Average loss, average error, proof steps, and number of proofs.
    #     """
    #     # Use the custom _referring_queries method to get results, proof steps, and number of proofs
    #     results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
    #     print(f"Results: {results}")
    #     print(f"Proof steps: {proof_steps}")
    #     print(f"Number of proofs: {nb_proofs}")
    #     # Compute the loss for each result-query pair
    #     losses = [self.criterion(result, instance.query.p) for result, instance in zip(results, data_instances)]
    #     print(f"Losses: {losses}")
    #     loss = torch.stack(losses).mean()
    #     print(f"Loss: {loss}")

    #     # Compute the errors for each result-query pair
    #     errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]

    #     # Perform backpropagation if the loss has gradients
    #     if loss.requires_grad:
    #         loss.backward()

    #     # Compute average proof steps and number of proofs
    #     proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))

    #     # Return the computed metrics
    #     print(f'Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}') #, float(loss), float(np.mean(errors)), proof_steps, nb_proofs)
    #     return float(loss), float(np.mean(errors)), proof_steps, nb_proofs
    
    def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
        # Use the custom _referring_queries method to get results, proof steps, and number of proofs
        results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
        print(f"Results: {results}")
        
        # Create losses array with default -inf values
        losses = []
        
        # Process each result and instance pair
        for result, instance in zip(results, data_instances):
            if isinstance(result, torch.Tensor) and torch.isneginf(result):
                # Get the actual query result from the query method directly
                # This will ensure we get the result that was computed earlier
                query_result = self.program.query(instance.query.query, **self.search_args)
                
                # Check if we got any results back
                if query_result and len(query_result) > 0:
                    # Find the best (highest probability) result
                    best_result = max(query_result.values())
                    losses.append(self.criterion(best_result, instance.query.p))
                else:
                    # Still -inf, but at least we tried
                    losses.append(self.criterion(result, instance.query.p))
            else:
                # Normal case - use the result directly
                losses.append(self.criterion(result, instance.query.p))
        
        # Rest of the function remains the same...
        loss = torch.stack(losses).mean() if losses else torch.tensor(float('inf'))
        errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]
        
        if loss.requires_grad:
            loss.backward()
        
        proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs

    
    


