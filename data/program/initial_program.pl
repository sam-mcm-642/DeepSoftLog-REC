% python run.py referring_expression train/config.yaml

ontology(~hyponym, ~dog, ~canine).
ontology(~hyponym, ~cat, ~feline).
ontology(~hyponym, ~dog, ~animal).
ontology(~hyponym, ~cat, ~animal).
ontology(~hyponym,~man,~person).
ontology(~hyponym,~woman,~person).
target(X) :- groundtruth(X, Y), object(X, Y). 
%type(X, Y) :- groundtruth(X, Z), object(X, Z).


expression(X, Y, Z) :- scene_graph(X, A, B), object(Y, A), object(Z, B).
% scene_graph(~rr1, A, B) :- scene_graph(~rr1, B, A).
% type(X, Y) :- ontology(Z, Y, X).


% ontology(~synonym,~person,~human).
% ontology(~partMeronym,~coat,~clothing).
% ontology(~partMeronym,~dress,~clothing).
% ontology(~partMeronym,~bag,~accessory).
% ontology(~partMeronym,~sneakers,~clothing).
% ontology(~partMeronym,~pants,~clothing).
% ontology(~partMeronym,~shirt,~clothing).
% ontology(~partMeronym,~jacket,~clothing).
% ontology(~partMeronym,~shoes,~clothing).
% ontology(~partMeronym,~glasses,~accessory).
% ontology(~partMeronym,~tree trunk,~tree).


% Dummy Instance
% groundtruth(~man, bbox1).
% object(~man, bbox1).
% target(X) :- groundtruth(X, Y), object(X, Y).
% type(X, Y) :- groundtruth(Y, Z), object(Y, Z).
% dummy(~person).
% ontology(~hyponym,~man,~person).
% type (X, Y) :- ontology(~synonym, X, Y).

% dummy(~wearing).

% type(X, Y) :- target(Y).


%to ensure that during training the model always tries to prove the query with
%the groundtruth being the target object

%Need a rule to ensure X is always ground to an object from the scene graph
%during inference
% scene_graph(~nextTo, bbox1, bbox2).
% object(~man, bbox1).
% object(~woman, bbox2).




%hard uniffy triplets from expression with identicial scene graph triplets

%%linking groundtruth label with target label in the query, either soft or hard unification


% type(~near, ~thing).
% %rules based on ontology
% object(~oobj1, X) :- object(Y, X), ontology(~rr1, ~oobj1, Y).


% %hasAttribute rules
% expression(~rr1, X, Y) :- scene_graph(hasAttribute, X, Y).
% is(X, Y) :- ontology(~rr1, X, Y).

