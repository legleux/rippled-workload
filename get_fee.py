import json
import ssl
from urllib import request
from urllib.error import HTTPError, URLError
from threading import Thread
from pathlib import Path
import time
import os
import sys

stop_at_ledger_size = 200

def healthy():
    url = "https://rippled:2459/health"
    ctx = ssl._create_unverified_context()
    req = request.Request(url)

    try:
        with request.urlopen(req, context=ctx) as resp:
            data = json.loads(resp.read())
            return True  # 200 means healthy

    except HTTPError as e:
        # rippled returns 503 with JSON when warning
        if e.code == 503:
            body = json.loads(e.read())
            info = body.get("info", {})
            # Our private net ALWAYS has 5 peers.
            # 503 + peers == 5 == totally fine.
            return info.get("peers") == 5

        # anything else is actually unhealthy
        return False

    except Exception:
        return False

def get_fee():
    url = "http://rippled:5005"
    try:
        req = request.Request(url, data=json.dumps({"method": "fee"}).encode())
        data = json.loads(request.urlopen(req, timeout=2).read())
        result = data["result"]
        return result
    except URLError:
        print("rippled not running")
        exit(1)

def server_info():
    url = "http://rippled:5005"
    try:
        req = request.Request(url, data=json.dumps({"method": "server_info"}).encode())
        data = json.loads(request.urlopen(req, timeout=2).read())
        result = data["result"]
        return result
    except URLError:
        print("rippled not running")
        exit(1)

def get_uptime():
    info = server_info()
    return info["info"].get("uptime")

def msg(uptime, lci, cls_, olf, els, cqs, msg, bar):
    print(bar)
    print("{:<25} = {:>6}".format("Uptime:", uptime))
    print("{:<25} = {:>6}".format("Ledger:", lci))
    print("{:<25} = {:>6}".format("Current ledger size:", cls_))
    print("{:<25} = {:>6}".format("Open ledger fee:", olf))
    print("{:<25} = {:>6}".format("Expected ledger size:", els))
    print("{:<25} = {:>6}".format("Current queue size:", cqs))
    for i in msg:
        print(i)

def main(fee_log_data):
    m = []
    if sys.argv[1] == "q":
        result = get_fee()
        print(json.dumps(result, indent=2))
        sys.exit(0)
    dots = []
    while True:
        result = get_fee()
        uptime = get_uptime()
        drops = result["drops"]
        lci = result["ledger_current_index"]
        cls_ = result["current_ledger_size"]
        olf = drops["open_ledger_fee"]
        els = result["expected_ledger_size"]
        cqs = result["current_queue_size"]
        if int(olf) > 10:
            fld = (lci, olf, cqs)
            if fld not in fee_log_data:
                fee_log_data.append(fld)
            if lci not in m:
                m.append(lci)
        if len(dots):
            if lci == llci:
                dots.append(".")
            else:
                llci = lci
                dots = ["."]
        else:
            llci = lci
            dots = ["."]
        bar = "".join(dots)
        msg(uptime, lci, cls_, olf, els, cqs, m, bar)

        time.sleep(0.05)
        os.system("clear")
        if int(cls_) == stop_at_ledger_size:
            print(f"Ledger breached {stop_at_ledger_size} at ledger {lci} after {uptime}s!")
            sys.exit(0)

def write_fee_log(fee_log_data):
    print("writing fee log")
    print(fee_log_data)
    fee_log = Path("fee.log")
    for ld in fee_log_data:
        lci, olf, cqs = ld
        with fee_log.open("a") as fl:
            fl.write("*****************\n")
            fl.write(f"Ledger: {lci}\n")
            fl.write(f"open ledger fee: {olf}\n")
            fl.write(f"current_queue_size: {cqs}\n")

from time import perf_counter
if __name__ == '__main__':
    fee_log = []
    timeout = 20
    start = perf_counter()

    while not healthy():
        now = perf_counter() - start
        if now > timeout:
            break
        time.sleep(1)
        print(f"waiting for rippled {timeout - now:.1f}s more...")
    try:
        main(fee_log)
    except KeyboardInterrupt:
        write_fee_log(fee_log)
        print('Quitting')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
