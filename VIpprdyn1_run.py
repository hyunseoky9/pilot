from VIpprdyn1 import *
import sys
if __name__ == "__main__":
    settingID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    vi = VIpprdyn1(settings={'settingID': settingID})
    policy, V = vi.value_iteration()