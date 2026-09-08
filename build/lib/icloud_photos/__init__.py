"""An assistant-friendly command line for an iCloud photo library.

pyicloud does the talking to Apple (login, session, listing, change feed,
downloads by size). This package adds what an assistant working through a
shell needs on top: a durable local catalogue that answers queries without
the network, a bounded on-disk cache of previews and originals with pinning
and eviction, named collections for projects, and a CLI whose every command
has --json output, bounded results and one-line errors.
"""

__version__ = "0.2.0"
