# -------- market --------
# The market data service, rebuilt: layout discovery, a binary sidecar index, and reads
# that touch only the bytes a caller asked for. Independent of sltp.data on purpose, so
# the old service can be retired without a rewrite elsewhere.
