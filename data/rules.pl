%strict unification rules 
target(X) :- groundtruth(X, Y), object(X, Y). 
%to ensure that during training the model always tries to prove the query with
%the groundtruth being the target object

%Need a rule to ensure X is always ground to an object from the scene graph
%during inference

expression(X, Y, Z) :- scene_graph(X, A, B), object(Y, A), object(Z, B).
%hard uniffy triplets from expression with identicial scene graph triplets

%%linking groundtruth label with target label in the query, either soft or hard unification
type(X, Y) :- ontology(~rr1, X, Y); ontology(~rr1, Y, X).


% %rules based on ontology
% object(~oobj1, X) :- object(Y, X), ontology(~rr1, ~oobj1, Y).


% %hasAttribute rules
% expression(~rr1, X, Y) :- scene_graph(hasAttribute, X, Y).
% is(X, Y) :- ontology(~rr1, X, Y).

