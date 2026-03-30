__all__ = [
    "LayerAdapterRegistry",
    "LayeredComputerUseMCP",
    "LayeredSession",
    "SessionRegistry",
    "SessionRuntimeConfig",
]


def __getattr__(name: str):
    if name in {"LayerAdapterRegistry", "SessionRuntimeConfig"}:
        from civStation.mcp.runtime import LayerAdapterRegistry, SessionRuntimeConfig

        return {
            "LayerAdapterRegistry": LayerAdapterRegistry,
            "SessionRuntimeConfig": SessionRuntimeConfig,
        }[name]
    if name in {"LayeredSession", "SessionRegistry"}:
        from civStation.mcp.session import LayeredSession, SessionRegistry

        return {
            "LayeredSession": LayeredSession,
            "SessionRegistry": SessionRegistry,
        }[name]
    if name == "LayeredComputerUseMCP":
        from civStation.mcp.server import LayeredComputerUseMCP

        return LayeredComputerUseMCP
    raise AttributeError(name)
