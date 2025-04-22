# 4.1.5.4. BHT (Branch History Table) submodule

BHT is implemented as a memory which is composed of **BHTDepth configuration parameter** entries. The lower address bits of the virtual address point to the memory entry.

When a branch instruction is resolved by the EX_STAGE module, the branch PC and the taken (or not taken) status information is stored in the Branch History Table.

The Branch History Table is a table of two-bit saturating counters that takes the virtual address of the current fetched instruction by the CACHE. It states whether the current branch request should be taken or not. The two bit counter is updated by the successive execution of the instructions as shown in the following figure.

When a branch instruction is pre-decoded by instr_scan submodule, the BHT valids whether the PC address is in the BHT and provides the taken or not prediction.

The BHT is never flushed.

| Signal | IO | Description | connection | Type |
|--------|-------|------------|------------|------|
| clk_i | in | Subsystem Clock | SUBSYSTEM | logic |
| rst_ni | in | Asynchronous reset active low | SUBSYSTEM | logic |
| vpc_i | in | Virtual PC | CACHE | logic[CVA6Cfg.VLEN-1:0] |
| bht_update_i | in | Update bht with resolved address | EXECUTE | bht_update_t |
| bht_prediction_o | out | Prediction from bht | FRONTEND | ariane_pkg::bht_prediction_t[CVA6Cfg.INSTR_PER_FETCH-1:0] |

Due to cv32a65x configuration, some ports are tied to a static value. These ports do not appear in the above table, they are listed below:

For any HW configuration:
- flush_bp_i input is tied to 0

As DebugEn = False:
- debug_mode_i input is tied to 0

## State Diagram

The two-bit counter state diagram shows four states: strongly not taken, weakly not taken, weakly taken, and strongly taken. Transitions occur based on the actual branch outcome.
