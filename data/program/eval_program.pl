%%EXPRESSION QUERY PROVING USING SCENE GRAPH FACTS

% ~ denotes a soft term, which can soft unify with another soft term

%%the key rule which links expression predicates from the query to scene graph facts
expression(X, Y, Z) :- scene_graph(X, A, B), object(Y, A), object(Z, B).


%to ensure that during training the model always tries to prove the query with
%the groundtruth being the target object
%%for training
% target(X) :- groundtruth(X, Y), object(X, Y). 


%%For evaluation - any object can bind to X
target(X) :- object(X, Y).


% type(X, Y). predicate is a builtin that returns true with probability equal to the soft unification score of X and Y


%%Sample scene graph facts for illustration purposes
%%Three types of facts (two during evalutation)
% scene_graph(~nextTo, bbox1, bbox2).
% object(~man, bbox1).
% object(~woman, bbox2).
% groundtruth(~man, bbox1). (only included for training)

%%sample query for this instance: target(X), type(X, ~man), expression(~nextTo, man, woman)


% scene_graph(~rr1, A, B) :- scene_graph(~rr1, B, A). (NOT IN USE)