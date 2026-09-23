# 13 — Session state — running clock and irreversibility

**What to build:** A stateful session wrapper over the engine's stateless `simulate()` core. Default mode is a running clock: time advances continuously at an adjustable rate, pausable. `tick()` advances one step. State is path-dependent and irreversible within a session — a well drawn down in simulated 2040 is still down in 2080 after the token is removed (removal stops the flow but does not undo the stock change). An explicit `reset()` returns to the initial condition. Sessions are the unit of state. The API exposes session lifecycle (create/tick/reset/destroy) over HTTP + WebSocket so the installation can drive it in real time.

**Blocked by:** 08 — Stock-and-flow state model, 11 — Token vocabulary

**Status:** ready-for-agent

- [ ] Session create/tick/pause/reset/destroy lifecycle is implemented
- [ ] `tick()` advances the stock model by one time step at the session's clock rate
- [ ] Removing a token stops its flow contribution but does not undo past stock changes
- [ ] Clock rate is adjustable (e.g. 1 simulated year per real second, configurable)
- [ ] `reset()` returns all stocks to their initial-condition values
- [ ] HTTP + WebSocket API exposes the full session lifecycle
- [ ] A paused session allows scrubbing via a horizon parameter
