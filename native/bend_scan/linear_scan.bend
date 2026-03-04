# Bend Linear Scan: h = a * h + b
#
# Bend is a massively parallel functional language that auto-parallelizes
# based on data dependencies. For linear scan, the sequential recurrence
# h = a*h + b is inherently sequential over timesteps, so Bend can only
# parallelize across batch and hidden dimensions.
#
# This is a learning exercise to understand Bend's execution model,
# NOT a practical GPU kernel implementation.

# Scan: compute all intermediate hidden states for one element
def linear_scan(h_state, a_seq, b_seq):
  match a_seq:
    case List/Cons:
      match b_seq:
        case List/Cons:
          h_new = (a_seq.head * h_state) + b_seq.head
          rest = linear_scan(h_new, a_seq.tail, b_seq.tail)
          return List/Cons { head: h_new, tail: rest }
        case List/Nil:
          return List/Nil
    case List/Nil:
      return List/Nil

# Test: batch=1, seq_len=4, hidden=2
#
# Hidden[0]: a=0.9, b=1.0, h0=0.0
#   t=0: 0.9*0.0 + 1.0 = 1.0
#   t=1: 0.9*1.0 + 1.0 = 1.9
#   t=2: 0.9*1.9 + 1.0 = 2.71
#   t=3: 0.9*2.71 + 1.0 = 3.439
#
# Hidden[1]: a=0.8, b=2.0, h0=0.0
#   t=0: 0.8*0.0 + 2.0 = 2.0
#   t=1: 0.8*2.0 + 2.0 = 3.6
#   t=2: 0.8*3.6 + 2.0 = 4.88
#   t=3: 0.8*4.88 + 2.0 = 5.904

def main():
  # Run two independent scans (Bend can parallelize these)
  h0 = linear_scan(0.0, [0.9, 0.9, 0.9, 0.9], [1.0, 1.0, 1.0, 1.0])
  h1 = linear_scan(0.0, [0.8, 0.8, 0.8, 0.8], [2.0, 2.0, 2.0, 2.0])
  return (h0, h1)
