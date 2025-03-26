

class VGMutator(object):
    pass


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


    def all_matches(self, term: Expr) -> Iterable[tuple[Clause, dict]]:
        predicate = term.get_predicate()
        print(f"\nDEBUG: Attempting to match term: {term}")
        print(f"DEBUG: Goal predicate: {predicate}")
        
        for db_clause in self.clauses:
            print(f"\nDEBUG: Examining clause: {db_clause}")
            
            # Handle both facts and rules correctly
            if isinstance(db_clause, Expr) and db_clause.functor != ':-':  # It's a fact
                db_head = db_clause
                print(f"DEBUG: Fact found - head: {db_head}")
                print(f"DEBUG: Fact predicate: {db_head.get_predicate()}")
                print(f"DEBUG: Does predicate match? {db_head.get_predicate() == predicate}")
            else:  # It's a rule
                if db_clause.functor == ':-':
                    db_head = db_clause.arguments[0]
                else:
                    db_head = db_clause.arguments[0]
                print(f"DEBUG: Rule found - head: {db_head}")
                print(f"DEBUG: Rule head predicate: {db_head.get_predicate()}")
                print(f"DEBUG: Does predicate match? {db_head.get_predicate() == predicate}")
            
            if db_head.get_predicate() == predicate:
                print(f"DEBUG: Attempting unification between:")
                print(f"DEBUG:   Term: {term}")
                print(f"DEBUG:   Head: {db_head}")
                result = self.mgu(term, db_head)
                print(f"DEBUG: Unification result: {result}")