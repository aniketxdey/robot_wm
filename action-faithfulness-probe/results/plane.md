# AFP Sensitivity Plane

Each model is a point `(gain_action, gain_instruction)` -- the per-channel dose-response gains (DESIGN Rev 4, Sec 2.2), in common units. Bottom-right = action-faithful (the goal); top-left = instruction-driven (the failure mode we hunt); top-right = reads both; near-origin = inert. The ASCII scatter normalizes each axis by the largest gain in the suite so the spread is visible.

| model | gain_action | gain_instruction | sens_action (share) | sens_instruction (share) | verdict |
|---|---|---|---|---|---|
| `mock-faithful` | 0.0543 | 0.0225 | 0.663 | 0.337 | action-faithful |
| `mock-balanced` | 0.0485 | 0.0637 | 0.406 | 0.594 | instruction-driven |
| `mock-instruction` | 0.0131 | 0.0647 | 0.185 | 0.815 | instruction-driven |

## Quadrant scatter (gains, normalized to the suite max)

```
  instr^
           2           |         1          
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
   --------------------+--------------------
                       |                    
                       |                    
                       |             0      
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
                       |                    
   0                                       >  action
```

x-axis = gain_action (0 left -> max right); y-axis = gain_instruction (max top -> 0 bottom). Markers are the model index in the legend below.

- `0` = `mock-faithful` (action-faithful)
- `1` = `mock-balanced` (instruction-driven)
- `2` = `mock-instruction` (instruction-driven)
