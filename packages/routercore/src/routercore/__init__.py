"""routercore: the router's Clean Architecture core (docs/adr/0002).

domain → application → adapters, each layer importing only the ones before it; services are the
frameworks & drivers ring and wire everything together.
"""
