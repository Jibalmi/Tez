# Preserved baseline implementation failure

The first baseline operation failed on its first validation batch. The newly written option
mask lacked the column dimension on the per-row option counts, so a 16-row batch with five
maximum options could not broadcast. The output prediction file is empty; no optimizer update
or continuation checkpoint was produced. This is an implementation defect, not a model-quality
or probability-parity result.

Executed source, the original registration, prior CPU input-preflight receipt, prior analysis
registration and the traceback remain here. The correction adds a tested batch readout helper
and amends the protocol before retrying in a new attempt directory. Model weights, data,
optimizer, selection rule, probability limits and resource bounds remain unchanged.
