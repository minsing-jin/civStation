__all__ = [
    "LayerAdapterRegistry",
    "LayeredComputerUseMCP",
    "LayeredSession",
    "SessionRegistry",
    "SessionRuntimeConfig",
]


def __getattr__(name: str):
    if name in {"LayerAdapterRegistry", "SessionRuntimeConfig"}:
        from computer_use_test.mcp.runtime import LayerAdapterRegistry, SessionRuntimeConfig

        return {
            "LayerAdapterRegistry": LayerAdapterRegistry,
            "SessionRuntimeConfig": SessionRuntimeConfig,
        }[name]
    if name in {"LayeredSession", "SessionRegistry"}:
        from computer_use_test.mcp.session import LayeredSession, SessionRegistry

        return {
            "LayeredSession": LayeredSession,
            "SessionRegistry": SessionRegistry,
        }[name]
    if name == "LayeredComputerUseMCP":
        from computer_use_test.mcp.server import LayeredComputerUseMCP

        return LayeredComputerUseMCP
    raise AttributeError(name)
