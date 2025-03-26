from deepsoftlog.algebraic_prover.terms.expression import Expr, Constant
from deepsoftlog.algebraic_prover.terms.variable import Variable
from deepsoftlog.logic.soft_term import SoftTerm
from deepsoftlog.data.query import Query




from typing import List, Union, Any


def query_to_prolog(string, **kwargs):
    # Parse the input string into individual expressions
    expressions = []
    
    # Split by commas and clean up whitespace, but be careful with nested commas inside parentheses
    parts = []
    current_part = ""
    paren_count = 0
    
    for char in string:
        if char == '(' or char == '[' or char == '{':
            paren_count += 1
            current_part += char
        elif char == ')' or char == ']' or char == '}':
            paren_count -= 1
            current_part += char
        elif char == ',' and paren_count == 0:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    # Parse each expression
    target_var = None
    for expr_str in parts:
        if expr_str.startswith("target("):
            # Extract the variable
            var_name = expr_str[len("target("):-1].strip()
            target_var = Variable(var_name)
            expressions.append(Expr("target", target_var))
        
        elif expr_str.startswith("type("):
            # Extract the variable and type
            content = expr_str[len("type("):-1].strip()
            # Use split with maxsplit to handle potential commas in the type name
            var_parts = content.split(',', 1)
            if len(var_parts) < 2:
                raise ValueError(f"Invalid type expression: {expr_str}")
                
            var_name = var_parts[0].strip()
            type_name = var_parts[1].strip()
            
            # Remove quotes if present
            if type_name.startswith("'") and type_name.endswith("'"):
                type_name = type_name[1:-1]
            if type_name.startswith('"') and type_name.endswith('"'):
                type_name = type_name[1:-1]
            
            # Ensure we're referencing the same variable
            var = target_var if target_var and var_name == target_var.name else Variable(var_name)
            expressions.append(Expr("type", var, SoftTerm(Constant(type_name))))
        
        elif expr_str.startswith("expression("):
            # Extract predicate, var and object
            content = expr_str[len("expression("):-1].strip()
            
            # Split carefully to handle potential commas in the arguments
            expr_parts = []
            part = ""
            nested_count = 0
            
            for char in content:
                if char == '(' or char == '[' or char == '{':
                    nested_count += 1
                    part += char
                elif char == ')' or char == ']' or char == '}':
                    nested_count -= 1
                    part += char
                elif char == ',' and nested_count == 0:
                    expr_parts.append(part.strip())
                    part = ""
                else:
                    part += char
            
            if part.strip():
                expr_parts.append(part.strip())
            
            if len(expr_parts) < 3:
                raise ValueError(f"Invalid expression: {expr_str}")
            
            predicate = expr_parts[0].strip()
            var_name = expr_parts[1].strip()
            obj_name = expr_parts[2].strip()
            
            # Handle negation if present
            negated = False
            if predicate.startswith("~"):
                negated = True
                predicate = predicate[1:]
            
            # Remove quotes if present
            if obj_name.startswith("'") and obj_name.endswith("'"):
                obj_name = obj_name[1:-1]
            if obj_name.startswith('"') and obj_name.endswith('"'):
                obj_name = obj_name[1:-1]
            
            # Create the expression
            var = target_var if target_var and var_name == target_var.name else SoftTerm(Constant(var_name))
            expr = Expr("expression", SoftTerm(Constant(predicate)), var, SoftTerm(Constant(obj_name)))
            
            # Apply negation if needed
            if negated:
                expr = ~expr
            
            expressions.append(expr)
    
    # Combine all expressions with the & operator
    if not expressions:
        return Query(None)  # Handle empty case
    
    combined_expr = expressions[0]
    for expr in expressions[1:]:
        combined_expr = combined_expr & expr
    
    return Query(combined_expr)


# Example usage
def example_usage():
    # Simple single expression query
    query1 = query_to_prolog("target(X)")
    print("Query 1:", query1)
    
    # Multiple expression query
    query2 = query_to_prolog("target(X), type(X, person), expression(nextTo, X, tree)")
    print("Query 2:", query2)
    
    # Query with different argument types
    query3 = query_to_prolog("expression(wearing, X, 'coat'), type(X, 'person')")
    print("Query 3:", query3)
    
    query4 = query_to_prolog("target(X), is(X, person), expression(wearing, X, coat), expression(nextTo, X, woman).")
    print("Query 4:", query4)

# Run examples
samp = Query(Expr("expression", SoftTerm(Constant("wearing")), SoftTerm(Constant("X")), SoftTerm(Constant("coat")))&Expr("type", SoftTerm(Constant("X")), SoftTerm(Constant("person"))))
print(samp)

example_usage()