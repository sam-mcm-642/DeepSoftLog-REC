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
        torch.autograd.set_detect_anomaly(True)
        
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
            loss.backward(retain_graph=True)
        
        proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        
        print("QUERY: ", instance.query)
        print(f"RESULTS: {result, instance in zip(results, data_instances)}")
        
        print(f'Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}') 
        
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs
    
    
    # def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
    #     torch.autograd.set_detect_anomaly(True)
        
    #     # Use the custom _referring_queries method to get results, proof steps, and number of proofs
    #     results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
    #     print(f"Results: {results}")
        
    #     # Create losses array with default -inf values
    #     losses = []
        
    #     # Process each result and instance pair
    #     for result, instance in zip(results, data_instances):
    #         if isinstance(result, torch.Tensor) and torch.isneginf(result):
    #             print(f"Result: {result}")
    #             print(f"Instance: {instance}")
    #             # Get the actual query result from the query method directly
    #             query_result = self.program.query(instance.query.query, **self.search_args)
    #             print(f"Query result: {query_result}")
    #             # Check if we got any results back
    #             if query_result and len(query_result) > 0:
    #                 # Find the best (highest probability) result
    #                 best_result = max(query_result.values())
    #                 # Clone the tensor to prevent in-place modifications
    #                 if isinstance(best_result, torch.Tensor) and best_result.requires_grad:
    #                     best_result = best_result.clone()
    #                 losses.append(self.criterion(best_result, instance.query.p))
    #             else:
    #                 # Still -inf, but at least we tried
    #                 losses.append(self.criterion(result, instance.query.p))
    #         else:
    #             # Normal case - use the result directly but clone if it's a tensor
    #             if isinstance(result, torch.Tensor) and result.requires_grad:
    #                 result_for_loss = result.clone()
    #             else:
    #                 result_for_loss = result
    #             losses.append(self.criterion(result_for_loss, instance.query.p))
        
    #     # Rest of the function remains the same...
    #     loss = torch.stack(losses).mean() if losses else torch.tensor(float('inf'))
    #     errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]
        
    #     if loss.requires_grad:
    #         loss.backward(retain_graph=True)
        
    #     proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        
    #     print("QUERY: ", instance.query)
    #     print(f"RESULTS: {result, instance in zip(results, data_instances)}")
        
    #     print(f'Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}') 
        
    #     return float(loss), float(np.mean(errors)), proof_steps, nb_proofs
    
    
    # def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
    #     # Enable anomaly detection for detailed error traces
    #     torch.autograd.set_detect_anomaly(True)
        
    #     print("\n=== DEBUG: Starting get_loss ===")
        
    #     # Get query results
    #     results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
    #     data_instances = list(data_instances)
    #     print(f"DEBUG: Got {len(results)} results")
        
    #     # Create losses array
    #     losses = []
        
    #     # Process each result and track tensor versions
    #     for i, (result, instance) in enumerate(zip(results, data_instances)):
    #         print(f"\nDEBUG: Processing item {i}")
            
    #         # Check result tensor before criterion
    #         if isinstance(result, torch.Tensor):
    #             print(f"DEBUG: Result is tensor with version {result._version}")
    #             print(f"DEBUG: requires_grad={result.requires_grad}, grad_fn={result.grad_fn}")
    #         else:
    #             print(f"DEBUG: Result is not a tensor: {type(result)}")
            
    #         # Compute loss term and track it
    #         loss_term = self.criterion(result, instance.query.p)
    #         losses.append(loss_term)
            
    #         # Check result tensor after criterion to detect in-place changes
    #         if isinstance(result, torch.Tensor):
    #             print(f"DEBUG: After criterion, result version={result._version}")
            
    #         # Also check the loss term
    #         if isinstance(loss_term, torch.Tensor):
    #             print(f"DEBUG: Loss term is tensor with version {loss_term._version}")
    #             print(f"DEBUG: requires_grad={loss_term.requires_grad}, grad_fn={loss_term.grad_fn}")
    #         else:
    #             print(f"DEBUG: Loss term is not a tensor: {type(loss_term)}")
        
    #     # Stack losses and check versions before and after
    #     print("\nDEBUG: Checking all loss versions before stack")
    #     for i, loss in enumerate(losses):
    #         if isinstance(loss, torch.Tensor):
    #             print(f"DEBUG: Loss {i} version={loss._version}")
        
    #     print("DEBUG: Stacking losses...")
    #     stacked = torch.stack(losses)
    #     print(f"DEBUG: Stacked shape={stacked.shape}")
        
    #     print("DEBUG: Checking loss versions after stack")
    #     for i, loss in enumerate(losses):
    #         if isinstance(loss, torch.Tensor):
    #             print(f"DEBUG: Loss {i} version={loss._version}")
        
    #     # Compute mean and check versions
    #     print("DEBUG: Computing mean...")
    #     loss = stacked.mean()
    #     print(f"DEBUG: Mean result={loss}")
        
    #     if isinstance(loss, torch.Tensor):
    #         print(f"DEBUG: Loss version={loss._version}, requires_grad={loss.requires_grad}")
        
    #     # Compute errors
    #     errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]
    #     error_mean = float(np.mean(errors)) if errors else 1.0
        
    #     # Backward pass - this is where the error will likely occur
    #     print("\nDEBUG: About to call backward()")
    #     if isinstance(loss, torch.Tensor) and loss.requires_grad:
    #         print("DEBUG: Calling loss.backward()")
    #         loss.backward()
    #         print("DEBUG: Backward completed successfully")
    #     else:
    #         print("DEBUG: Loss doesn't require grad, skipping backward")
        
    #     # Return values
    #     proof_steps_mean = float(np.mean(proof_steps))
    #     nb_proofs_mean = float(np.mean(nb_proofs))
    #     loss_value = float(loss) if isinstance(loss, torch.Tensor) else loss
        
    #     print("=== DEBUG: Finished get_loss ===")
        
    #     return loss_value, error_mean, proof_steps_mean, nb_proofs_mean
    
    # def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
    #     torch.autograd.set_detect_anomaly(True)
        
    #     # Use the custom _referring_queries method to get results, proof steps, and number of proofs
    #     results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
    #     print(f"Results: {results}")
        
    #     # Create losses array with default -inf values
    #     losses = []
        
    #     # Process each result and instance pair
    #     for result, instance in zip(results, data_instances):
    #         if isinstance(result, torch.Tensor) and torch.isneginf(result):
    #             print(f"Result: {result}")
    #             print(f"Instance: {instance}")
    #             # Get the actual query result from the query method directly
    #             query_result = self.program.query(instance.query.query, **self.search_args)
    #             print(f"Query result: {query_result}")
    #             # Check if we got any results back
    #             if query_result and len(query_result) > 0:
    #                 # Find the best (highest probability) result
    #                 best_result = max(query_result.values())
    #                 # Clone to avoid in-place operations affecting the original
    #                 if isinstance(best_result, torch.Tensor) and best_result.requires_grad:
    #                     best_result = best_result.detach().clone().requires_grad_()
    #                 losses.append(self.criterion(best_result, instance.query.p))
    #             else:
    #                 # Skip -inf results
    #                 continue
    #         else:
    #             # Normal case - clone to avoid in-place modifications
    #             if isinstance(result, torch.Tensor) and result.requires_grad:
    #                 # Detach, clone and re-attach to computation graph
    #                 result_for_loss = result.detach().clone().requires_grad_()
    #                 losses.append(self.criterion(result_for_loss, instance.query.p))
    #             else:
    #                 losses.append(self.criterion(result, instance.query.p))
        
    #     # Handle case with no valid losses
    #     if not losses:
    #         return float('inf'), 1.0, float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        
    #     # Calculate mean and errors
    #     loss = torch.stack(losses).mean()
    #     errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]
        
    #     # Backward pass
    #     if loss.requires_grad:
    #         # No need for retain_graph=True since we've detached tensors
    #         loss.backward()
        
    #     proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        
    #     print("QUERY: ", instance.query)
    #     print(f"RESULTS: {result, instance in zip(results, data_instances)}")
        
    #     print(f'Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}') 
        
    #     return float(loss), float(np.mean(errors)), proof_steps, nb_proofs