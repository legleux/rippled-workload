from fastapi import Request

from workload.workload_core import Workload


def get_workload(request: Request) -> Workload:
    return request.app.state.workload
