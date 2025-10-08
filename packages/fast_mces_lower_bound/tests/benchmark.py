# import polars as pl
import fast_mces_lower_bound
from time import perf_counter
import os
import threading
import time
import resource

smiles_list = [
    "c1ccc2cc3ccccc3cc2c1", # anthracene
    "c1ccc2cc3cCCcc3cc2c1", # anthracene with two middle carbons aliphatic
    "CC(NC)CC1=CC=C(OCO2)C2=C1", # mdma
    "CCCCCc1cc(c2c(c1)OC([C@H]3[C@H]2C=C(CC3)C)(C)C)O", #THC
    r"Oc1c(c(O)cc(c1)CCCCC)[C@@H]2\C=C(/CC[C@H]2\C(=C)C)C", #CBD
    "O=C4[C@@H]5Oc1c2c(ccc1OC)C[C@H]3N(CC[C@]25[C@@]3(O)CC4)C", # oxycodone
    "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", # morphine
    "Nc1cc(C)ccc1",  # m-methylaniline (meta-methyl aniline)
    "Nc1ccc(C)cc1",  # p-methylaniline (para-methyl aniline)
    "CN(C)C(Cc1ccccc1)",  # methamphetamine (N-methylamphetamine)
    "CC(Cc1ccc(C)cc1)N",  # 4-methylamphetamine (methyl on the benzene ring, para)
]*1000

# Lightweight /proc monitor that samples this process' RSS (kB).
# This includes memory used by any C/C++ extensions loaded into this process.
def _read_proc_status_value(key: str):
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])  # value is in kB
                        except ValueError:
                            return None
    except Exception:
        return None
    return None

def _monitor_rss(running_event: threading.Event, holder: dict, interval: float = 0.05):
    while running_event.is_set():
        v = _read_proc_status_value("VmRSS:")
        if v is not None and v > holder["val"]:
            holder["val"] = v
        # also check VmHWM if present (peak)
        h = _read_proc_status_value("VmHWM:")
        if h is not None and h > holder.get("hwm", 0):
            holder["hwm"] = h
        time.sleep(interval)

monitor_running = threading.Event()
monitor_running.set()
max_rss = {"val": 0, "hwm": 0}
monitor_thread = threading.Thread(target=_monitor_rss, args=(monitor_running, max_rss), daemon=True)
monitor_thread.start()

start = perf_counter()
result = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
end = perf_counter()

# stop monitor and collect final numbers
monitor_running.clear()
monitor_thread.join(timeout=1.0)

# get peak RSS reported by the OS for this process (kB). Includes C/C++ allocations.
rusage = resource.getrusage(resource.RUSAGE_SELF)
ru_maxrss_kb = getattr(rusage, "ru_maxrss", 0)  # typically in kB on Linux

# read final /proc/self/status values
vmrss_kb = _read_proc_status_value("VmRSS:") or 0
vmhwm_kb = _read_proc_status_value("VmHWM:") or 0

print(f"Calculated symmetric distance matrix for {len(smiles_list)} SMILES in {end - start:.4f} seconds.")
print(f"Peak RSS observed by monitor thread: {max_rss['val']} kB ({max_rss['val']/1024:.2f} MB)")
print(f"Peak VmHWM observed by monitor thread: {max_rss.get('hwm', 0)} kB ({max_rss.get('hwm',0)/1024:.2f} MB)")
print(f"ru_maxrss (getrusage): {ru_maxrss_kb} kB ({ru_maxrss_kb/1024:.2f} MB)")
print(f"VmHWM (proc status peak): {vmhwm_kb} kB ({vmhwm_kb/1024:.2f} MB)")
print(f"VmRSS (current resident set): {vmrss_kb} kB ({vmrss_kb/1024:.2f} MB)")