from typing import cast

from relbench.tasks import register_task
from relbench.base import BaseTask

from .bpi2017 import CaseRNextEvent, CaseRNextTime, CaseRRemainingTime
from .bpi2019 import POItemNextEvent, POItemNextTime, POItemRemainingTime
from .container_logistics import ContainerNextEvent, ContainerNextTime, ContainerRemainingTime

# BPI 2017 - Case_R
register_task("bpi2017", "next_event_cases", cast(BaseTask, CaseRNextEvent))
register_task("bpi2017", "next_time_cases", cast(BaseTask, CaseRNextTime))
register_task("bpi2017", "remaining_time_cases", cast(BaseTask, CaseRRemainingTime))

# BPI 2019 - PO Item
register_task("bpi2019", "next_event_po_items", cast(BaseTask, POItemNextEvent))
register_task("bpi2019", "next_time_po_items", cast(BaseTask, POItemNextTime))
register_task("bpi2019", "remaining_time_po_items", cast(BaseTask, POItemRemainingTime))

# Container Logistics - Container
register_task("container_logistics", "next_event_containers", cast(BaseTask, ContainerNextEvent))
register_task("container_logistics", "next_time_containers", cast(BaseTask, ContainerNextTime))
register_task("container_logistics", "remaining_time_containers", cast(BaseTask, ContainerRemainingTime))
