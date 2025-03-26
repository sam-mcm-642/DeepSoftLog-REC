from gentopia import PromptTemplate


prompt = PromptTemplate(
    input_variables=["instruction", "fewshot"],
    template="""You are an expert in extracting structured representations from natural language referring expressions. Your task is to convert a given referring expression into a set of Prolog-like facts that describe the target object and its relationships.

Output Format:
	•	Target Object: target(X) (where X represents the unknown target object).
	•	Type of Target Object: type(X, ~<category>) (infer the general category, e.g., ~person, ~object, ~animal).
	•	Relationships: expression(~<predicate>, <subject>, <object>).
	•	The predicate (e.g., ~sittingOn, ~nextTo, ~on) represents the relationship inferred from the referring expression.
	•	The subject is either X (the target object) or another object in the description.
	•	Objects (e.g., ~bench, ~person, ~skateboard) should be extracted as general concepts.



##Examples##

{fewshot}


##Your Task##

INPUT: {instruction}
OUTPUT: 
"""
)