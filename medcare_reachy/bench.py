"""Time the vision models on this robot and estimate the landmark frame rate it can keep up.

Over SSH on the robot:  /venvs/apps_venv/bin/python -m medcare_reachy.bench
"""

from medcare_reachy import models
from medcare_reachy.bridge import vision_bench

if __name__ == "__main__":
    vision_bench.main(models.verify_models())
