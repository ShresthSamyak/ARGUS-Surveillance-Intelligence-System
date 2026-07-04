"""The slow path — LocateAnything-3B behind a VLMProvider interface (§5.7, v0.2).

System 2 to the perception fast path's System 1: deliberative (0.5-2 s/query),
advisory, crash-isolated. Its outputs enter the entity graph as evidence with a
source tag + confidence — never blocking a tick, never directly overwriting
fast-path state. If this process dies, Argus degrades to v0.1 semantics.

Phase 1 ships the VLMProvider interface + job queue as stubs (no model loaded)
so Phase 2 plugs in the model instead of retrofitting.
"""
