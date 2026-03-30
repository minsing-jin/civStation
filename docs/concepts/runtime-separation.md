# Runtime Separation

The README does not describe CivStation as one loop with one responsibility. It describes a system where different kinds of work run in different lanes.

## The Three Runtime Lanes

### Background runtime

This lane owns work that benefits from running beside the action loop:

- context observation
- turn tracking
- strategy refresh
- background reasoning

### Main-thread action runtime

This is the lane that must stay deterministic and interruptible:

- route the current screen
- plan the primitive action
- execute the action on the game window

### HitL runtime

This lane keeps control outside the action thread:

- dashboard
- WebSocket control
- relay and mobile controller
- strategy and lifecycle directives

## Why This Matters

If you mix all of that into one blocking loop:

- expensive reasoning stalls action execution
- interruption becomes fragile
- strategy refinement arrives too late
- external control becomes an afterthought

Runtime separation is what makes CivStation feel operable instead of merely clever.
