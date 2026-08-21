"""
Shared 5-stage pipeline building blocks for the threaded NPU runners.

The two threaded drivers - run_pipelined.py (§ single vs pipelined) and
benchmark.py (§ python vs C++) - used to carry their own byte-identical copies
of the model-A and model-B inference stages, differing ONLY in the metric-dict
key names they accumulate into (prep_person_ms/… vs prep_A_ms/…). Those stages
live here once, parameterised by key name, so each driver keeps its own reader /
crop / collector (where the experiments genuinely differ - clip set vs loop
budget, render vs telemetry, CSV schema) without duplicating the inference core.
"""
import time

import common


def stage_model1(models, q_in, q_out, k_prep, k_infer, k_post, trace_label=None):
    """Model A (person) inference + output extraction. `k_*` are the metric-dict
    keys this driver accumulates the set-tensor / invoke / extract times into.
    `trace_label` (optional): if set and the item carries a "trace" dict, record
    this stage's wall-clock [start,end] for the pipeline Gantt."""
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None)
            return
        t_stage0 = time.perf_counter()
        with common.Timer() as t_set:
            models.person.set_tensor(models.person_input["index"], item.pop("input_a"))
        with common.Timer() as t_infer:
            models.person.invoke()
        with common.Timer() as t_extract:
            item["person_raw"] = common.extract_person_outputs(models.person)
        item["times"][k_prep] += t_set.elapsed_ms
        item["times"][k_infer] += t_infer.elapsed_ms
        item["times"][k_post] += t_extract.elapsed_ms
        if trace_label and "trace" in item:
            item["trace"][trace_label] = (t_stage0, time.perf_counter())
        q_out.put(item)


def stage_model2(models, q_in, q_out, k_prep, k_infer, k_post, trace_label=None):
    """Model B (PPE) inference per cropped person + output extraction. Sequential
    per crop - a model-B interpreter pool is impossible (the QNN delegate
    segfaults creating a 2nd HTP context for the same model)."""
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None)
            return
        t_stage0 = time.perf_counter()
        item["ppe_raw"] = []
        for input_b in item.pop("inputs_b"):
            with common.Timer() as t_set:
                models.ppe.set_tensor(models.ppe_input["index"], input_b)
            with common.Timer() as t_infer:
                models.ppe.invoke()
            with common.Timer() as t_extract:
                item["ppe_raw"].append(common.extract_ppe_outputs(models.ppe))
            item["times"][k_prep] += t_set.elapsed_ms
            item["times"][k_infer] += t_infer.elapsed_ms
            item["times"][k_post] += t_extract.elapsed_ms
        if trace_label and "trace" in item:
            item["trace"][trace_label] = (t_stage0, time.perf_counter())
        q_out.put(item)
