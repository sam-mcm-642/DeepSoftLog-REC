from deepsoftlog.logic.soft_unify import soft_mgu
from deepsoftlog.algebraic_prover.terms.expression import Expr

# Create test terms
oobj1 = Expr("~", Expr("oobj1"))  # Assuming this is how ~oobj1 is represented
man = Expr("man")

# Try to soft-unify
result = soft_mgu(oobj1, man, store, metric)
print(f"Soft unification test result: {result}")
