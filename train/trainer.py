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
                print(f"Referring result: {result}")

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
                print(f"Result: {result}")
                print(f"Instance: {instance}")
                # Get the actual query result from the query method directly
                # This will ensure we get the result that was computed earlier
                query_result = self.program.query(instance.query.query, **self.search_args)
                print(f"Query result: {query_result}")
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
        
        print("QUERY: ", instance.query)
        print(f"RESULTS: {result, instance in zip(results, data_instances)}")
        
        print(f'Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}') 
        
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs