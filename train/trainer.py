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

            print(f"Before conversion - Result type: {type(result)}, value: {result}")
            # Ensure result is a tensor if it isn't already
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
                print(f"Referring result: {result}")
            
            print(f"After conversion - Result type: {type(result)}, value: {result}, requires_grad: {result.requires_grad if hasattr(result, 'requires_grad') else 'N/A'}")

            # Yield the result, proof steps, and number of proofs
            yield result, proof_steps, nb_proofs
        
    def visualize_key_embeddings(self):
        """Visualize key embeddings and their relationships"""
        print("\n=== Key Embedding Relationships ===")
        
        # Get embeddings for key terms
        try:
            store = self.program.store
            
            # Common terms in your dataset
            key_terms = ['man', 'person', 'dog', 'animal']
            embeddings = {}
            
            for term in key_terms:
                if term in store.constant_embeddings:
                    embeddings[term] = store.constant_embeddings[term].detach()
            
            # Print similarities between key pairs
            for term1 in key_terms:
                for term2 in key_terms:
                    if term1 != term2 and term1 in embeddings and term2 in embeddings:
                        # Calculate cosine similarity
                        e1 = embeddings[term1]
                        e2 = embeddings[term2]
                        sim = torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()
                        print(f"Similarity({term1}, {term2}) = {sim:.4f}")
        
        except Exception as e:
            print(f"Error in embedding visualization: {e}")  
            
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
        """
        Compute loss with proper groundtruth object handling.
        """
        data_instances = list(data_instances)
        results = []
        proof_steps_list = []
        nb_proofs_list = []
        
        for idx, instance in enumerate(data_instances):
            print(f"Processing instance {idx+1}/{len(data_instances)}")
            
            # Update clauses
            self.program.update_clauses(instance)
            
            # Get query and groundtruth
            if not isinstance(instance.query, Query):
                instance.query = query_to_prolog(instance.query)
            
            query = instance.query.query
            groundtruth = instance.target[0] if hasattr(instance, 'target') and instance.target else None
            print(f"Query: {query}, Groundtruth: {groundtruth}")
            
            # Call the program with groundtruth object
            result, steps, proofs = self.program(
                query, 
                groundtruth_object=groundtruth,
                **self.search_args
            )
            
            # Ensure result is a tensor with requires_grad=True
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, requires_grad=True)
            elif not result.requires_grad:
                result = result.detach().clone().requires_grad_(True)
            
            results.append(result)
            proof_steps_list.append(steps)
            nb_proofs_list.append(proofs)
        
        # Compute losses
        losses = []
        errors = []
        
        for result, instance in zip(results, data_instances):
            # Compute loss term
            loss_term = self.criterion(result, instance.query.p)
            if not isinstance(loss_term, torch.Tensor):
                loss_term = torch.tensor(loss_term, requires_grad=True)
            elif not loss_term.requires_grad:
                loss_term = loss_term.detach().clone().requires_grad_(True)
            
            losses.append(loss_term)
            errors.append(instance.query.error_with(result))
        
        # Stack losses
        loss = torch.stack(losses).mean()
        
        # Backward pass
        if loss.requires_grad:
            loss.backward(retain_graph=True)
         
        self.visualize_key_embeddings()
        return float(loss), float(np.mean(errors)), float(np.mean(proof_steps_list)), float(np.mean(nb_proofs_list))
        
    
    # def get_loss(self, data_instances: Iterable[DatasetInstance]) -> tuple[float, float, float, float]:
    #     torch.autograd.set_detect_anomaly(True)
        
    #     # Use the custom _referring_queries method to get results, proof steps, and number of proofs
    #     results, proof_steps, nb_proofs = tuple(zip(*self._referring_queries(data_instances)))
    #     print(f"Results: {results}")
        
    #     # Create losses array with default -inf values
    #     losses = []
        
    #     # Add this to get_loss after getting results and before calculating losses:
    #     print("\n=== Examining proof results ===")
    #     for i, result in enumerate(results):
    #         print(f"Result {i}: {result}")
    #         print(f"Result type: {type(result)}")
    #         if isinstance(result, torch.Tensor):
    #             print(f"Result requires_grad: {result.requires_grad}")
    #             # Check if there are soft facts associated with this result
    #             if hasattr(result, 'pos_facts'):
    #                 print(f"Result has {len(result.pos_facts)} soft facts")
    #                 for fact in result.pos_facts:
    #                     if hasattr(fact, 'get_log_probability'):
    #                         log_prob = fact.get_log_probability()
    #                         print(f"  Soft fact log_prob: {log_prob}")
    #                         print(f"  log_prob type: {type(log_prob)}")
    #                         if isinstance(log_prob, torch.Tensor):
    #                             print(f"  log_prob requires_grad: {log_prob.requires_grad}")
        
        
    #     # Process each result and instance pair
    #     for result, instance in zip(results, data_instances):
    #         if isinstance(result, torch.Tensor) and torch.isneginf(result):
    #             print(f"Result: {result}")
    #             print(f"Instance: {instance}")
    #             # Get the actual query result from the query method directly
    #             # This will ensure we get the result that was computed earlier
    #             query_result = self.program.query(instance.query.query, **self.search_args)
    #             print(f"Query result: {query_result}")
    #             for key, value in query_result.items():
    #                 print(f"{key} ({type(key).__name__}): {type(value).__name__}")
    #             # Check if we got any results back
    #             if query_result and len(query_result) > 0:
    #                 # Find the best (highest probability) result
    #                 best_result = torch.tensor(max(query_result.values()), requires_grad=True)
    #                 print(self.criterion(best_result, instance.query.p))
    #                 losses.append(self.criterion(best_result, instance.query.p))
    #                 print(type(self.criterion(best_result, instance.query.p)))
    #             else:
    #                 # Still -inf, but at least we tried
    #                 losses.append(self.criterion(result, instance.query.p))
    #         else:
    #             # Normal case - use the result directly
    #             print("Using result directly")
    #             losses.append(self.criterion(result, instance.query.p))
        
    #     # Rest of the function remains the same...
    #     for i, loss in enumerate(losses):
    #         print(f"Loss {i}: type={type(loss)}, requires_grad={loss.requires_grad if hasattr(loss, 'requires_grad') else 'N/A'}")
    #     loss = torch.stack(losses).mean() if losses else torch.tensor(float('inf'))
    #     errors = [instance.query.error_with(result) for result, instance in zip(results, data_instances)]
        
    #     # Also add to get_loss after loss calculation to check the gradients:
    #     if loss.requires_grad:
    #         print("\n=== Performing backward pass ===")
    #         loss.backward(retain_graph=True)  # Use retain_graph=True to avoid errors
            
    #         # Check gradients
    #         print("\n=== Checking gradients after backward ===")
    #         has_grad = False
    #         for name, param in self.program.store.named_parameters():
    #             if param.grad is not None and param.grad.norm() > 0:
    #                 has_grad = True
    #                 print(f"Parameter {name} has gradient norm: {param.grad.norm().item()}")
            
    #         if not has_grad:
    #             print("No parameters have gradients!")
        
    #     if loss.requires_grad:
    #         loss.backward(retain_graph=True)
        
    #     proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        
    #     print("QUERY: ", instance.query)
    #     print(f"RESULTS: {result, instance in zip(results, data_instances)}")
        

    #     print(f"Final Loss: {loss}, Error: {np.mean(errors)}, Proof Steps: {proof_steps}, Nb Proofs: {nb_proofs}")
    #     return float(loss), float(np.mean(errors)), proof_steps, nb_proofs

    
    
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
    #                 result_for_loss = resultch().clone().requires_grad_()
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