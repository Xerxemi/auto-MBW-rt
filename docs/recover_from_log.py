import numpy as np

raw_log = """
[0] hyper_score (Bayesian Optimization):  12%| | 31/250 [7:10:18<50:39:56, 832.86s/it, best_iter=0, bes


Results: 'hyper_score'  
   Best score: 0.44875530845332806  
   Best parameter set:
      '0'  : 0.35  
      '1'  : 0.05  
      '2'  : 1.0  
      '3'  : 0.7  
      '4'  : 0.35  
      '5'  : 0.75  
      '6'  : 0.7
      '7'  : 1.0
      '8'  : 0.95
      '9'  : 0.65
      '10' : 0.9
      '11' : 0.05
      '12' : 0.45
      '13' : 1.0
      '14' : 0.4
      '15' : 0.4
      '16' : 0.3
      '17' : 1.0
      '18' : 0.5
      '19' : 0.7
      '20' : 0.8
      '21' : 0.6
      '22' : 0.95
      '23' : 0.55
      '24' : 0.45
      '25' : 0.2
      '26' : 0.2
   Best iteration: 0

   Random seed: 60156508

   Evaluation time   : 25650.57901096344 sec    [99.74 %]
   Optimization time : 66.75476264953613 sec    [0.26 %]
   Iteration time    : 25717.333773612976 sec    [102.87 sec/iter]

2023-11-05 15:40:31,671 - AutoMBWrt - INFO - merge completed.
2023-11-05 15:40:32,474 - AutoMBWrt - INFO - injection disabled (hopefully).
"""

print("Recover Weights from console log: ")
print(raw_log)
print("log = weights = slider")

# sliders = sliders_in + [sl_M_00] + [sl_OUT] + sliders_out + [sl_TIME_EMBED] = sl_ALL_nat

# sl_ALL_nat = [*sl_INPUT, *sl_MID, sl_OUT, *sl_OUTPUT, sl_TIME_EMBED]
# sl_ALL = [*sl_INPUT, *sl_MID, *sl_OUTPUT, sl_TIME_EMBED, sl_OUT]

print("Repo1: https://github.com/Xerxemi/auto-MBW-rt")
print("Repo2: https://github.com/Xynonners/sd-webui-runtime-block-merge/blob/master/scripts/runtime_block_merge.py")

def on_save_checkpoint(*weights, ):
    current_weights_nat = weights[:27]
    weights_output_recipe = weights[27:]
    print("current_weights_nat:")
    print(current_weights_nat)
    print("weights_output_recipe:")
    print(f"{','.join([str(w) for w in weights_output_recipe])}\n")
    print("Which is [*sl_INPUT, *sl_MID, *sl_OUTPUT, sl_OUT, sl_TIME_EMBED]")
    print("However in code sl_ALL (Repo2): ")
    print("sl_ALL = [*sl_INPUT, *sl_MID, *sl_OUTPUT, sl_TIME_EMBED, sl_OUT]")

def main(weights):
    weights_s = weights.copy()
    out = weights_s.pop(13)
    time_embed = weights_s.pop(-1)
    weights_s.append(out)
    weights_s.append(time_embed)
    print("weights:")
    print(weights)
    print("weights_s:")
    print(weights_s) # sliders_in + [sl_M_00] + sliders_out + [sl_OUT] + [sl_TIME_EMBED] = !sl_ALL
    on_save_checkpoint(*weights,*weights_s) # Merge is fine, receipe is bad.

sliders = list(range(0,27))

print("(Repo1): sliders = sliders_in + [sl_M_00] + [sl_OUT] + sliders_out + [sl_TIME_EMBED] = sl_ALL_nat")
print(sliders)

main(sliders)

print("Due to the receipe bug, you should swap OUT and TIME_EMBED on UI if you are reading the receipe.")

actual_receipe = "0.35,0.05,1.0,0.7,0.35,0.75,0.7,1.0,0.95,0.65,0.9,0.05,0.45,0.4,0.4,0.3,1.0,0.5,0.7,0.8,0.6,0.95,0.55,0.45,0.2,1.0,0.2"
print("Verify with Actual receipe: ")
print(actual_receipe)